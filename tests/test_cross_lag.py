# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Unit tests for the VG16 cross-lag helpers (issues #113 and #242).

The lag source is assigned per complete ``(subject, age)`` administration
wave: every row of a wave receives the same prior distinct-age source, the
source state advances only after a whole wave is processed, and where a source
wave carries several understood measurements the largest count is selected.
The regression tests here pin the three failure modes of the row-by-row walk
this replaced: the ``[12, 24, 24]`` parallel-form pattern, row-order
dependence, and multiple source-form rows (issue #242).
"""

import numpy as np
import pandas as pd
import pytest

from vocab_growth.models.common_bivariate_re import (
    _compute_prev_wave_lag,
    _validate_cross_lag,
    compute_prev_wave_lag,
    cross_lag_audit_frame,
)
from vocab_growth.models.likelihood_utils import (
    LAG_ZERO_CLIP,
    LAG_ZERO_CONTINUITY,
)

N_TRIALS = 800


def logit(p):
    return np.log(p) - np.log(1 - p)


def _synthetic_df():
    """Rows cover the cases the lag source must handle, in a known row order.

    - subject 0: three waves, all with understood -> waves 2/3 have a prior source
    - subject 1: a single wave -> no prior source
    - subject 2: first wave lacks understood -> its later wave has no valid source
    - subject 3: middle wave lacks understood -> the source skips it to an earlier wave
    """
    return pd.DataFrame(
        {
            "subject_code": [0, 0, 0, 1, 2, 2, 3, 3, 3],
            "age": [12, 24, 36, 18, 12, 24, 12, 24, 36],
            "understood": [100, 200, 300, 150, np.nan, 180, 120, np.nan, 360],
        }
    )


def test_compute_prev_wave_lag_identifies_prior_understood_wave():
    prev_idx, has_lag_f, _ = _compute_prev_wave_lag(_synthetic_df(), N_TRIALS)

    # Only rows with a valid earlier understood wave carry a lag.
    np.testing.assert_array_equal(has_lag_f, [0, 1, 1, 0, 0, 0, 0, 1, 1])
    # prev_idx points at that earlier wave (row 8's source skips the NaN row 7 -> row 6).
    assert prev_idx[1] == 0
    assert prev_idx[2] == 1
    assert prev_idx[7] == 6
    assert prev_idx[8] == 6


def test_compute_prev_wave_lag_logit_of_prior_understood():
    _, has_lag_f, y_u_prev_logit = _compute_prev_wave_lag(_synthetic_df(), N_TRIALS)

    # Non-lagged rows are 0; lagged rows carry the logit of the prior wave's proportion.
    np.testing.assert_allclose(
        y_u_prev_logit[[1, 2, 7, 8]],
        [logit(100 / 800), logit(200 / 800), logit(120 / 800), logit(120 / 800)],
        rtol=1e-6,
    )
    np.testing.assert_array_equal(y_u_prev_logit[[0, 3, 4, 5, 6]], 0.0)


def test_compute_prev_wave_lag_ignores_same_age_duplicates():
    """A same-age duplicate is part of the first wave, never a source for it.

    Both age-12 rows belong to one wave, so neither receives a lag; the age-24
    row's source is the age-12 wave with the largest count selected (150, row 1).
    """
    df = pd.DataFrame(
        {
            "subject_code": [0, 0, 0],
            "age": [12, 12, 24],
            "understood": [100, 150, 300],
        }
    )

    prev_idx, has_lag_f, y_u_prev_logit = _compute_prev_wave_lag(df, N_TRIALS)

    np.testing.assert_array_equal(has_lag_f, [0, 0, 1])
    assert prev_idx[2] == 1
    assert y_u_prev_logit[1] == 0.0
    np.testing.assert_allclose(y_u_prev_logit[2], logit(150 / 800), rtol=1e-6)


@pytest.mark.parametrize("understood_first", [True, False])
def test_compute_prev_wave_lag_parallel_form_rows_share_the_wave_source(
    understood_first,
):
    """The ``[12, 24, 24]`` pattern that broke the row-by-row walk (issue #242).

    A child measured at 12 months, then given two checklist forms at 24 months
    — one carrying understood, one spoken-only. Both age-24 rows form one wave
    and must receive the age-12 source, whichever of them appears first: the
    old walk advanced its state after the understood row, so the parallel
    spoken-only row received a lag only under one of the two tie orders.
    """
    if understood_first:
        understood = [130.0, 260.0, np.nan]
    else:
        understood = [130.0, np.nan, 260.0]
    df = pd.DataFrame(
        {
            "subject_code": [0, 0, 0],
            "age": [12, 24, 24],
            "understood": understood,
        }
    )

    prev_idx, has_lag_f, y_u_prev_logit = _compute_prev_wave_lag(df, N_TRIALS)

    np.testing.assert_array_equal(has_lag_f, [0, 1, 1])
    np.testing.assert_array_equal(prev_idx[[1, 2]], [0, 0])
    np.testing.assert_allclose(y_u_prev_logit[[1, 2]], logit(130 / 800), rtol=1e-6)


def test_compute_prev_wave_lag_selects_largest_count_at_the_source_wave():
    """Multiple source-form rows: the largest understood count is the source.

    Two forms at the source wave carry different understood counts (a shorter
    form right-truncates the shared inventory), so the selection rule — the
    largest count, the least-truncated measurement — must pick 300 whichever
    row order the frame arrives in.
    """
    df = pd.DataFrame(
        {
            "subject_code": [0, 0, 0],
            "age": [18, 18, 30],
            "understood": [300.0, 200.0, 400.0],
        }
    )

    for order in ([0, 1, 2], [1, 0, 2], [2, 1, 0]):
        shuffled = df.iloc[order].reset_index(drop=True)
        _, has_lag_f, y_u_prev_logit = _compute_prev_wave_lag(shuffled, N_TRIALS)
        target = shuffled.index[shuffled["age"] == 30][0]
        np.testing.assert_array_equal(has_lag_f[target], 1.0)
        np.testing.assert_allclose(y_u_prev_logit[target], logit(300 / 800), rtol=1e-6)


def test_compute_prev_wave_lag_is_row_order_invariant():
    """Permuting the input rows permutes the outputs and changes nothing else.

    The frame combines every structure that trips per-row state: parallel
    same-age forms with and without understood, a wave with two understood
    measurements, skipped no-comprehension waves, and single-wave children.
    The old walk failed this under almost any shuffle (issue #242).
    """
    df = pd.DataFrame(
        {
            "subject_code": [0, 0, 0, 0, 1, 1, 1, 2, 3, 3, 3, 3],
            "age": [12, 24, 24, 36, 10, 10, 20, 30, 14, 26, 26, 40],
            "understood": [
                90.0, 210.0, np.nan, 350.0,
                60.0, 110.0, 240.0,
                140.0,
                np.nan, 180.0, 175.0, np.nan,
            ],
        }
    )
    n = len(df)
    age = df["age"].to_numpy(float)
    prev0, has0, ylog0 = _compute_prev_wave_lag(df, N_TRIALS)
    source_age0 = np.where(has0 > 0, age[prev0], np.nan)

    rng = np.random.default_rng(20260823)
    for _ in range(10):
        perm = rng.permutation(n)
        shuffled = df.iloc[perm].reset_index(drop=True)
        prev_p, has_p, ylog_p = _compute_prev_wave_lag(shuffled, N_TRIALS)
        # Row k of the shuffle is original row perm[k]; every per-row output
        # must follow it, and the source's identity (its age and its count via
        # the logit) must be the same wave whichever row index carries it.
        np.testing.assert_array_equal(has_p, has0[perm])
        np.testing.assert_allclose(ylog_p, ylog0[perm], rtol=1e-12)
        source_age_p = np.where(
            has_p > 0, shuffled["age"].to_numpy(float)[prev_p], np.nan
        )
        np.testing.assert_allclose(source_age_p, source_age0[perm], equal_nan=True)
        # And the source always lies strictly earlier, for the same child.
        lagged = has_p > 0
        subj_p = shuffled["subject_code"].to_numpy(int)
        age_p = shuffled["age"].to_numpy(float)
        assert (subj_p[prev_p[lagged]] == subj_p[lagged]).all()
        assert (age_p[prev_p[lagged]] < age_p[lagged]).all()


def test_compute_prev_wave_lag_array_api_matches_dataframe_adapter():
    df = _synthetic_df()
    from_arrays = compute_prev_wave_lag(
        df["subject_code"].to_numpy(int),
        df["age"].to_numpy(float),
        df["understood"].to_numpy(float),
        N_TRIALS,
    )
    from_frame = _compute_prev_wave_lag(df, N_TRIALS)
    for a, b in zip(from_arrays, from_frame, strict=True):
        np.testing.assert_array_equal(a, b)


def test_cross_lag_audit_frame_records_support_gaps_forms_and_branch():
    df = pd.DataFrame(
        {
            "subject_code": [0, 0, 0, 1, 1],
            "age": [12, 24, 24, 18, 30],
            "understood": [0.0, 250.0, np.nan, 120.0, np.nan],
            "study": ["s1", "s1", "s1", "s2", "s2"],
            "survey_vocab_max": [396.0, 810.0, 680.0, 810.0, 810.0],
        }
    )
    prev_idx, has_lag_f, _ = _compute_prev_wave_lag(df, N_TRIALS)
    # Rows 1, 2 and 4 carry a spoken observation: 1 conditional, 2 and 4 marginal.
    audit = cross_lag_audit_frame(
        df,
        prev_idx,
        has_lag_f,
        spoken_indices=np.array([1, 2, 4]),
        spoken_is_conditional=np.array([True, False, False]),
    )

    # Rows 1 and 2 (subject 0's second wave) and row 4 (subject 1's) have a source.
    np.testing.assert_array_equal(audit["row"].to_numpy(), [1, 2, 4])
    np.testing.assert_array_equal(audit["source_row"].to_numpy(), [0, 0, 3])
    np.testing.assert_array_equal(audit["gap_months"].to_numpy(), [12.0, 12.0, 12.0])
    np.testing.assert_array_equal(
        audit["source_understood_zero"].to_numpy(), [True, True, False]
    )
    np.testing.assert_array_equal(
        audit["source_wave_understood_measurements"].to_numpy(), [1, 1, 1]
    )
    np.testing.assert_array_equal(
        audit["spoken_branch"].to_numpy(), ["conditional", "marginal", "marginal"]
    )
    np.testing.assert_array_equal(audit["study"].to_numpy(), ["s1", "s1", "s2"])
    np.testing.assert_array_equal(
        audit["source_form_ceiling"].to_numpy(), [396.0, 396.0, 810.0]
    )
    np.testing.assert_array_equal(
        audit["form_ceiling_changed"].to_numpy(), [True, True, False]
    )


def test_cross_lag_audit_frame_tolerates_minimal_frames():
    """Injected simulation frames carry no study or form-ceiling columns."""
    df = pd.DataFrame(
        {
            "subject_code": [0, 0],
            "age": [12, 24],
            "understood": [100.0, 200.0],
        }
    )
    prev_idx, has_lag_f, _ = _compute_prev_wave_lag(df, N_TRIALS)
    audit = cross_lag_audit_frame(
        df,
        prev_idx,
        has_lag_f,
        spoken_indices=np.array([1]),
        spoken_is_conditional=np.array([True]),
    )
    assert list(audit["row"]) == [1]
    assert "study" not in audit.columns
    assert "form_ceiling_changed" not in audit.columns


@pytest.mark.parametrize("baseline", ["population", "within"])
def test_validate_cross_lag_accepts_valid_config(baseline):
    _validate_cross_lag(baseline, use_subject_re_u=True)  # no raise


def test_validate_cross_lag_rejects_unknown_baseline():
    with pytest.raises(ValueError, match="lag_baseline"):
        _validate_cross_lag("bogus", use_subject_re_u=True)


@pytest.mark.parametrize("baseline", ["population", "within"])
def test_validate_cross_lag_requires_subject_re_u(baseline):
    with pytest.raises(ValueError, match="use_subject_re_u"):
        _validate_cross_lag(baseline, use_subject_re_u=False)


# --- Gap ceiling and zero-count handling (issue #242) --------------------------


def test_gap_ceiling_drops_the_lag_but_keeps_the_row():
    """A gap ceiling must not remove observations, only their lag.

    The row still enters both likelihoods; it simply stops informing
    ``beta_lag``. Removing the row instead would confound the gap question with
    a sample-size change.
    """
    df = _synthetic_df()
    base_idx, base_lag, _ = _compute_prev_wave_lag(df, N_TRIALS)
    idx, lag, logits = compute_prev_wave_lag(
        df["subject_code"].to_numpy(int),
        df["age"].to_numpy(float),
        df["understood"].to_numpy(float),
        N_TRIALS,
        max_gap_months=12.0,
    )
    # Subject 0's waves are 12 months apart and survive; subject 3's row 8 has a
    # 24-month gap to its source at row 6 and loses its lag.
    np.testing.assert_array_equal(base_lag, [0, 1, 1, 0, 0, 0, 0, 1, 1])
    np.testing.assert_array_equal(lag, [0, 1, 1, 0, 0, 0, 0, 1, 0])
    assert len(lag) == len(base_lag)          # no row removed
    assert logits[8] == 0.0                   # and its predictor is neutralised


def test_gap_ceiling_never_changes_which_wave_is_the_source():
    """The ceiling gates whether a source is used, not which one is chosen.

    If it were applied while walking the waves, a row just over the ceiling
    could fall back to an *earlier* source and so acquire a longer gap than the
    one that was rejected.
    """
    df = _synthetic_df()
    base_idx, base_lag, _ = _compute_prev_wave_lag(df, N_TRIALS)
    idx, lag, _ = compute_prev_wave_lag(
        df["subject_code"].to_numpy(int),
        df["age"].to_numpy(float),
        df["understood"].to_numpy(float),
        N_TRIALS,
        max_gap_months=12.0,
    )
    kept = lag > 0
    np.testing.assert_array_equal(idx[kept], base_idx[kept])


def test_no_ceiling_reproduces_the_historical_lag_exactly():
    df = _synthetic_df()
    base = _compute_prev_wave_lag(df, N_TRIALS)
    same = compute_prev_wave_lag(
        df["subject_code"].to_numpy(int),
        df["age"].to_numpy(float),
        df["understood"].to_numpy(float),
        N_TRIALS,
        max_gap_months=None,
        zero_handling=LAG_ZERO_CLIP,
    )
    for a, b in zip(base, same, strict=True):
        np.testing.assert_array_equal(a, b)


def test_continuity_correction_moves_a_zero_source_off_the_clip():
    """A zero source sits at logit(1e-4) under the clip, a value set by the floor
    rather than by the data — identical whether the form had 810 items or 396."""
    df = pd.DataFrame({
        "subject_code": [0, 0],
        "age": [12, 24],
        "understood": [0.0, 200.0],
    })
    args = (
        df["subject_code"].to_numpy(int),
        df["age"].to_numpy(float),
        df["understood"].to_numpy(float),
        N_TRIALS,
    )
    _, _, clipped = compute_prev_wave_lag(*args, zero_handling=LAG_ZERO_CLIP)
    _, _, corrected = compute_prev_wave_lag(*args, zero_handling=LAG_ZERO_CONTINUITY)

    assert clipped[1] == pytest.approx(logit(1e-4), rel=1e-9)
    assert corrected[1] == pytest.approx(logit(0.5 / (N_TRIALS + 1)), rel=1e-9)
    # The correction is a boundary treatment, not a rescaling: it is well inside
    # the clip, and the two differ by nearly two logits on this row.
    assert corrected[1] > clipped[1] + 1.5


def test_continuity_correction_barely_moves_a_non_boundary_source():
    df = pd.DataFrame({
        "subject_code": [0, 0],
        "age": [12, 24],
        "understood": [400.0, 500.0],
    })
    args = (
        df["subject_code"].to_numpy(int),
        df["age"].to_numpy(float),
        df["understood"].to_numpy(float),
        N_TRIALS,
    )
    _, _, clipped = compute_prev_wave_lag(*args, zero_handling=LAG_ZERO_CLIP)
    _, _, corrected = compute_prev_wave_lag(*args, zero_handling=LAG_ZERO_CONTINUITY)
    assert abs(corrected[1] - clipped[1]) < 0.01


def test_unknown_zero_handling_is_rejected():
    df = _synthetic_df()
    with pytest.raises(ValueError, match="lag_zero_handling"):
        compute_prev_wave_lag(
            df["subject_code"].to_numpy(int),
            df["age"].to_numpy(float),
            df["understood"].to_numpy(float),
            N_TRIALS,
            zero_handling="halve-it",
        )
