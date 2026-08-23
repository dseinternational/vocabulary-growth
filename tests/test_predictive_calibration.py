# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for quantitative posterior-predictive calibration summaries."""

import numpy as np
import pandas as pd
import pytest

from vocab_growth.models.calibration import (
    DEFAULT_INTERVAL_PROBS,
    predictive_calibration_table,
)


def test_predictive_calibration_reports_overall_and_age_bands():
    observed = np.array([0, 2, 4, 6])
    predictive = np.array(
        [
            [0, 0, 1, 1],
            [1, 2, 2, 3],
            [3, 4, 4, 5],
            [5, 6, 6, 7],
        ]
    )
    ages = np.array([8, 10, 14, 18])

    table = predictive_calibration_table(observed, predictive, ages)

    assert set(table["age_band_months"]) == {"all", "[0, 12)", "[12, 24)"}
    # Asserted against the convention rather than literals: the tabulated levels
    # are tied to vocab_growth.intervals so predictive coverage is reported at the
    # same widths as every credible interval in the report.
    assert set(table["interval_probability"]) == set(DEFAULT_INTERVAL_PROBS)
    overall = table[table["age_band_months"] == "all"]
    assert (overall["empirical_coverage"] == 1.0).all()
    assert (overall["n_observations"] == 4).all()


def test_predictive_calibration_rejects_misaligned_inputs():
    with pytest.raises(ValueError, match="row-aligned"):
        predictive_calibration_table(
            observed=np.array([1]),
            predictive=np.ones((2, 4)),
            ages=np.array([10]),
        )


def test_predictive_calibration_chunking_preserves_results():
    observed = np.array([0, 2, 4, 6])
    predictive = np.array(
        [[0, 0, 1, 1], [1, 2, 2, 3], [3, 4, 4, 5], [5, 6, 6, 7]],
        dtype=np.int16,
    )
    ages = np.array([8, 10, 14, 18])

    expected = predictive_calibration_table(observed, predictive, ages)
    chunked = predictive_calibration_table(
        observed,
        predictive,
        ages,
        observation_chunk_size=1,
    )

    pd.testing.assert_frame_equal(chunked, expected)


def _written_table(levels=(0.5, 0.8, 0.9)):
    """A calibration table in the written schema, with chosen nominal levels."""
    rows = []
    for outcome in ("understood", "spoken"):
        for band, n in (("all", 40), ("[0, 12)", 10), ("[12, 24)", 25), ("[108, 120)", 5)):
            for level in levels:
                rows.append(
                    {
                        "outcome": outcome,
                        "age_band_months": band,
                        "n_observations": n,
                        "observed_mean": 100.0,
                        "predictive_mean": 101.0,
                        "mean_error": 1.0,
                        "observed_zero_rate": 0.0,
                        "predictive_zero_rate": 0.0,
                        "mid_pit_mean": 0.5,
                        "mid_pit_variance": 0.06,
                        "mid_pit_extreme_rate": 0.05,
                        "interval_probability": level,
                        "empirical_coverage": 0.9,
                        "mean_interval_width": 50.0,
                    }
                )
    return pd.DataFrame(rows)


def test_default_levels_follow_the_reporting_convention():
    """Predictive coverage must be tabulated at the widths the report leads with.

    Otherwise the report can only quote coverage at some unrelated round number
    while every credible interval beside it is an 89% one.
    """
    from vocab_growth import intervals
    from vocab_growth.models.calibration import DEFAULT_INTERVAL_PROBS

    assert intervals.INNER_CI_PROB in DEFAULT_INTERVAL_PROBS
    assert intervals.DEFAULT_CI_PROB in DEFAULT_INTERVAL_PROBS


def test_nominal_level_resolves_from_the_table_not_from_an_assumption():
    """An older table tabulated 0.90; a current one tabulates 0.89.

    The reporting helpers must read whichever is present so an older fit's table
    is labelled with the level it actually holds rather than mislabelled.
    """
    from vocab_growth.models.calibration import nominal_level

    assert nominal_level(_written_table(levels=(0.5, 0.8, 0.9))) == 0.9
    assert nominal_level(_written_table(levels=(0.5, 0.8, 0.89))) == 0.89
    # An explicit target still wins, so the inner interval can be reported too.
    assert nominal_level(_written_table(), target=0.5) == 0.5


def test_nominal_level_rejects_a_frame_that_is_not_a_calibration_table():
    from vocab_growth.models.calibration import nominal_level

    with pytest.raises(ValueError, match="calibration table|no interval"):
        nominal_level(pd.DataFrame({"outcome": ["understood"]}))


def test_overall_calibration_selects_the_pooled_rows_only():
    from vocab_growth.models.calibration import overall_calibration

    frame, level = overall_calibration(_written_table())

    assert level == 0.9
    assert set(frame["age_band_months"]) == {"all"}
    assert list(frame["outcome"]) == ["understood", "spoken"]


def test_calibration_by_age_sorts_bands_numerically():
    """``[108, 120)`` must follow ``[12, 24)``, not precede it.

    Age bands are strings, so a lexical sort would put the 9-year band second and
    make the age trend unreadable.
    """
    from vocab_growth.models.calibration import calibration_by_age

    frame, _ = calibration_by_age(_written_table())

    understood = frame[frame["outcome"] == "understood"]
    assert list(understood["age_band_months"]) == ["[0, 12)", "[12, 24)", "[108, 120)"]
    assert "all" not in set(frame["age_band_months"])


def test_format_calibration_labels_and_rounds_the_reported_columns():
    from vocab_growth.models.calibration import format_calibration, overall_calibration

    frame, _ = overall_calibration(_written_table())
    display = format_calibration(frame)

    assert list(display.columns)[:3] == ["Outcome", "Age band (mo)", "n"]
    assert "Coverage" in display.columns and "PIT variance" in display.columns
    # Internal column names must not leak into a reported table.
    assert not {"mid_pit_mean", "empirical_coverage"} & set(display.columns)
    assert display["n"].dtype.kind in "iu"


def test_render_calibration_section_explains_an_absent_table(tmp_path, capsys):
    """A fit predating the calibration table must not render an empty section."""
    from vocab_growth.models.calibration import render_calibration_section

    render_calibration_section(str(tmp_path))

    out = capsys.readouterr().out
    assert "No calibration table" in out
    assert "refit" in out


def test_render_calibration_section_reports_the_level_and_both_tables(tmp_path, capsys):
    from vocab_growth.models.calibration import render_calibration_section

    _written_table().to_csv(
        tmp_path / "posterior_predictive_calibration.csv", index=False
    )
    render_calibration_section(str(tmp_path))

    out = capsys.readouterr().out
    assert "**90%**" in out  # the level actually tabulated, not an assumed one
    assert "0.083" in out  # the uniform-PIT reference value
    assert "pooled over ages" in out and "by age band" in out.lower()
    assert "| Outcome" in out  # rendered as a markdown table


def test_render_calibration_section_carries_the_in_sample_caveat(tmp_path, capsys):
    """The caveat must travel with the numbers, not live only in the report.

    A reader of a single model's page would otherwise take over-coverage as
    evidence that the reported intervals are conservative for a new child, which
    an in-sample posterior predictive check cannot support.
    """
    from vocab_growth.models.calibration import render_calibration_section

    _written_table().to_csv(
        tmp_path / "posterior_predictive_calibration.csv", index=False
    )
    render_calibration_section(str(tmp_path))

    out = capsys.readouterr().out
    assert "in-sample" in out.lower()
    assert "not* evidence" in out or "not evidence" in out
    assert "LOO" in out  # points at the out-of-sample counterpart


def test_discrete_reference_matches_theory_for_calibrated_bernoulli():
    """Perfectly calibrated Bernoulli(0.5): mid-PIT variance 0.0625, not 1/12.

    The canonical counter-example to the continuous-uniform reference (#234):
    the observed mid-PIT variance must sit close to the model-implied
    ``expected_mid_pit_variance`` — well below 1/12 — with no miscalibration
    anywhere, and the equal-tailed discrete interval covers with probability 1
    against a 0.89 nominal level, which ``expected_coverage`` must record.
    """
    rng = np.random.default_rng(7)
    predictive = rng.integers(0, 2, size=(500, 4000))
    observed = rng.integers(0, 2, size=500)
    ages = np.full(500, 10.0)

    table = predictive_calibration_table(observed, predictive, ages)
    overall = table[table["age_band_months"] == "all"]

    assert overall["expected_mid_pit_variance"].iloc[0] == pytest.approx(
        0.0625, abs=0.002
    )
    assert overall["mid_pit_variance"].iloc[0] == pytest.approx(
        overall["expected_mid_pit_variance"].iloc[0], abs=0.01
    )
    # Discreteness alone puts every level's expected coverage at 1 here.
    assert (overall["expected_coverage"] > 0.99).all()


def test_expected_coverage_tracks_nominal_for_a_spread_out_predictive():
    """With mass spread over many counts the discrete corrections shrink."""
    rng = np.random.default_rng(3)
    predictive = rng.binomial(400, 0.5, size=(300, 4000))
    observed = rng.binomial(400, 0.5, size=300)
    ages = np.full(300, 10.0)

    table = predictive_calibration_table(observed, predictive, ages)
    overall = table[table["age_band_months"] == "all"]

    assert overall["expected_mid_pit_variance"].iloc[0] == pytest.approx(
        1.0 / 12.0, abs=0.003
    )
    for _, row in overall.iterrows():
        # Discrete intervals cover at least their nominal mass (the interval
        # endpoints are atoms), with the excess shrinking as the mass spreads —
        # here at most a few points of probability.
        assert (
            row["interval_probability"] - 1e-9
            <= row["expected_coverage"]
            <= row["interval_probability"] + 0.05
        )


def test_render_calibration_section_uses_the_discrete_reference(tmp_path, capsys):
    from vocab_growth.models.calibration import render_calibration_section

    observed = np.array([0, 2, 4, 6])
    predictive = np.array(
        [[0, 0, 1, 1], [1, 2, 2, 3], [3, 4, 4, 5], [5, 6, 6, 7]]
    )
    table = predictive_calibration_table(
        observed, predictive, np.array([8.0, 10.0, 14.0, 18.0])
    )
    table.insert(0, "outcome", "words")
    table.to_csv(tmp_path / "posterior_predictive_calibration.csv", index=False)

    render_calibration_section(str(tmp_path))
    out = capsys.readouterr().out
    assert "Calibrated coverage" in out
    assert "Calibrated PIT variance" in out
    assert "not against the nominal level" in out


def test_render_calibration_section_explains_a_legacy_table(tmp_path, capsys):
    """A table written before the discrete reference existed still renders honestly."""
    from vocab_growth.models.calibration import render_calibration_section

    observed = np.array([0, 2, 4, 6])
    predictive = np.array(
        [[0, 0, 1, 1], [1, 2, 2, 3], [3, 4, 4, 5], [5, 6, 6, 7]]
    )
    table = predictive_calibration_table(
        observed, predictive, np.array([8.0, 10.0, 14.0, 18.0])
    )
    table = table.drop(columns=["expected_mid_pit_variance", "expected_coverage"])
    table.insert(0, "outcome", "words")
    table.to_csv(tmp_path / "posterior_predictive_calibration.csv", index=False)

    render_calibration_section(str(tmp_path))
    out = capsys.readouterr().out
    assert "predates the calibrated discrete reference" in out
    assert "Calibrated coverage" not in out
