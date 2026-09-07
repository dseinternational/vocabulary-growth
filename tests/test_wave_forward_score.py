# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The wave-forward sequential validation target (issue #242 item 5, #289 3.8).

VG16's understood PSIS-LOO is suppressed because the lag predictor for a child's
later wave embeds their earlier understood count, so leaving one administration
out of the likelihood does not leave it out of the model. The registered
replacement is a grouped forward-chaining score.

What is pinned here is the part that decides whether the score means anything:
which rows leave the likelihood, which rows are scored, and that both agree with
the model's own lag rule. The fold fits themselves are the same
``build_model_re`` path ``kfold_loso.py`` uses and are exercised by that script's
own coverage; nothing here samples.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from vocab_growth.models.definitions import MODEL_REGISTRY

# Loaded by path rather than through `sys.path`, following
# `tests/test_compact_traces.py`: `scripts/` is not a package, and putting it on
# the path for the whole session would let a script shadow a module for every
# other test in the run.
_SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "wave_forward_score.py"
_SPEC = importlib.util.spec_from_file_location("wave_forward_score_script", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
wf = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = wf
_SPEC.loader.exec_module(wf)


def _frame(rows) -> pd.DataFrame:
    """``(subject_code, age, understood, spoken)`` rows as a prepared frame."""
    df = pd.DataFrame(rows, columns=["subject_code", "age", "understood", "spoken"])
    df["study_code"] = 0
    df["study"] = "s"
    return df


# --- Waves --------------------------------------------------------------------


def test_two_forms_at_one_age_are_one_wave_not_two():
    """The wave definition issue #242 settled, applied here.

    If two same-day rows counted as two waves, the second would be scored as a
    "later wave" predicted from the first — a same-age prediction the lag rule
    explicitly refuses to make.
    """
    frame = _frame([
        (1, 12.0, 50, 10),
        (1, 12.0, 60, 12),   # second form, same day
        (1, 24.0, 200, 80),
    ])
    np.testing.assert_array_equal(wf.wave_index(frame), [0, 0, 1])


def test_the_wave_index_does_not_depend_on_row_order():
    rows = [(1, 24.0, 200, 80), (2, 12.0, 30, 5), (1, 12.0, 50, 10), (2, 30.0, 90, 40)]

    def waves_by_child_and_age(frame):
        return dict(
            zip(
                zip(frame["subject_code"], frame["age"], strict=True),
                wf.wave_index(frame),
                strict=True,
            )
        )

    base = waves_by_child_and_age(_frame(rows))
    shuffled = waves_by_child_and_age(_frame([rows[i] for i in (3, 0, 2, 1)]))
    assert base == shuffled


def test_each_child_restarts_at_zero():
    frame = _frame([
        (1, 12.0, 50, 10), (1, 24.0, 200, 80), (1, 36.0, 400, 300),
        (2, 18.0, 70, 20), (2, 30.0, 150, 60),
    ])
    np.testing.assert_array_equal(wf.wave_index(frame), [0, 1, 2, 0, 1])


# --- What leaves the likelihood, and what is scored ---------------------------


FOLD_FRAME = _frame([
    (1, 12.0, 50, 10), (1, 24.0, 200, 80),
    (2, 18.0, 70, 20), (2, 30.0, 150, 60), (2, 42.0, 300, 200),
    (3, 15.0, 40, 8),
])


def test_later_waves_holds_out_the_fold_childs_later_rows_only():
    """Their first wave must stay in training: it is the lag source.

    A held-out row's predictor is built from that row's earlier wave. Removing
    the earlier wave too would make the score condition on data the model never
    saw, which is the ``child`` unit's question, not this one's.
    """
    mask = wf.holdout_mask(FOLD_FRAME, np.array([1, 2]), unit="later-waves")
    np.testing.assert_array_equal(mask, [False, True, False, True, True, False])


def test_child_holds_out_every_row_of_the_fold_child():
    mask = wf.holdout_mask(FOLD_FRAME, np.array([1, 2]), unit="child")
    np.testing.assert_array_equal(mask, [True, True, True, True, True, False])


def test_the_scored_set_is_the_same_under_both_units():
    """The units differ in what is *trained on*, not in what is scored.

    If they differed in both, the two runs would not be comparable at all.
    """
    fold = np.array([1, 2])
    np.testing.assert_array_equal(wf.scored_rows(FOLD_FRAME, fold), [1, 3, 4])


def test_a_first_wave_is_never_scored():
    """It carries no lag, so both arms give it the identical density.

    Scoring it would add a row that cannot move the difference but does move the
    paired standard error.
    """
    fold = np.array([3])  # a singleton child
    assert len(wf.scored_rows(FOLD_FRAME, fold)) == 0


def test_every_child_is_in_exactly_one_fold():
    frame, _ = _registered_frame()
    folds, _subj = wf.stratified_subject_folds(frame, K=5)
    allocated = np.concatenate(folds)
    assert len(allocated) == len(set(allocated.tolist()))
    assert set(allocated.tolist()) == set(frame["subject_code"].unique().tolist())


# --- Agreement with the model's own lag rule ----------------------------------


def _registered_frame():
    from vocab_growth.models.common_bivariate_re import (
        build_bivariate_re_analysis_frame,
    )

    definition = MODEL_REGISTRY["vg16"]
    frame, _meta = build_bivariate_re_analysis_frame(definition)
    return frame, definition


@pytest.mark.slow
def test_no_first_wave_carries_a_lag_on_the_registered_frame():
    """The two wave notions must coincide, or the scored set is the wrong one.

    ``wave_index`` walks the frame here; ``prev_wave_lag_for_frame`` walks it in
    the engine. They are separate implementations of "a child's administration
    waves in age order", and a disagreement would show up as a first wave with a
    lag source — a same-age or backwards prediction.
    """
    frame, definition = _registered_frame()
    lagged, _source = wf.lag_source(frame, definition)
    assert not lagged[wf.wave_index(frame) == 0].any()


@pytest.mark.slow
def test_the_lagged_scored_rows_are_the_coefficients_own_support():
    """Cross-check against the count the #242 audit reached independently.

    The audit says the coefficient rests on 473 administrations. That is exactly
    the lagged rows carrying a spoken observation, so if this script's scored set
    disagreed with it, one of the two would be describing a different model.
    """
    frame, definition = _registered_frame()
    lagged, _source = wf.lag_source(frame, definition)
    supporting = int((lagged & frame["spoken"].notna().to_numpy()).sum())
    assert supporting == 473


@pytest.mark.slow
def test_the_audits_unlagged_later_wave_count_is_reproduced():
    """153 later rows whose every earlier wave lacked comprehension.

    The third number the available-case audit reached independently. Together
    with the 975 first waves it accounts for every row the coefficient cannot
    reach, so agreement here means the two are describing the same frame.
    """
    frame, definition = _registered_frame()
    lagged, _source = wf.lag_source(frame, definition)
    later = wf.wave_index(frame) > 0
    assert int((later & ~lagged).sum()) == 153
    assert int((~later).sum()) == 975


# --- Which scored rows stand on training data ---------------------------------


def test_a_second_wave_is_predicted_from_training_and_a_third_is_not():
    """Under ``later-waves`` the split is real and has to be visible.

    A second wave's lag source is the child's first, which stays in training. A
    *third* wave's source is the second, which does not — so its predictor
    conditions on a row the likelihood never saw. Still forward chaining, but a
    weaker claim, and the headline restriction takes the strict side.
    """
    frame = _frame([
        (1, 12.0, 50, 10),   # wave 0 — trained on
        (1, 24.0, 200, 80),  # wave 1 — source is wave 0
        (1, 36.0, 400, 300),  # wave 2 — source is wave 1, which is held out
    ])
    definition = MODEL_REGISTRY["vg16"]
    lagged, source = wf.lag_source(frame, definition)
    mask = wf.holdout_mask(frame, np.array([1]), unit="later-waves")
    clean = wf.source_in_training(lagged, source, mask)

    np.testing.assert_array_equal(lagged, [False, True, True])
    np.testing.assert_array_equal(clean, [False, True, False])


def test_holding_out_the_whole_child_leaves_no_source_in_training():
    """Which is the stricter unit's defining property, stated as a test."""
    frame = _frame([
        (1, 12.0, 50, 10), (1, 24.0, 200, 80), (1, 36.0, 400, 300),
    ])
    definition = MODEL_REGISTRY["vg16"]
    lagged, source = wf.lag_source(frame, definition)
    mask = wf.holdout_mask(frame, np.array([1]), unit="child")
    assert not wf.source_in_training(lagged, source, mask).any()


# --- The control arm ----------------------------------------------------------


def test_the_control_differs_in_the_coefficient_and_the_output_directory_only():
    """It exists to isolate one boolean.

    Comparing against VG10 instead would confound the coefficient with every
    other field the two definitions do not share.
    """
    import dataclasses

    definition = MODEL_REGISTRY["vg16"]
    control = wf.control_definition(definition)
    changed = {
        f.name
        for f in dataclasses.fields(definition)
        if getattr(definition, f.name) != getattr(control, f.name)
    }
    assert changed == {"use_cross_lag", "config_name"}
    assert control.use_cross_lag is False
    assert control.config_name.endswith(wf.CONTROL_SUFFIX)


def test_only_cross_lag_models_can_be_scored():
    """Without a coefficient to remove, the two arms are the same model."""
    assert "vg16" in wf.CROSS_LAG_MODELS
    assert "vg10" not in wf.CROSS_LAG_MODELS
    for key in wf.CROSS_LAG_MODELS:
        assert MODEL_REGISTRY[key].use_cross_lag


# --- The paired difference ----------------------------------------------------


def _wide(has_lag_flags, lag, control, clean=None):
    return pd.DataFrame({
        "has_lag": has_lag_flags,
        "source_in_training": has_lag_flags if clean is None else clean,
        "elpd_spoken_lag": lag,
        "elpd_spoken_control": control,
    })


def test_unlagged_rows_change_the_standard_error_but_not_the_difference():
    """Which is why the headline restricts to the rows a coefficient can move.

    A row with no lag enters both arms identically, so its difference is exactly
    zero: it leaves the total alone and shrinks the per-row spread the standard
    error is built from, making the comparison look more precise than its
    evidence.
    """
    lagged = _wide([True] * 4, [-1.0, -2.0, -3.0, -4.0], [-1.5, -2.5, -2.5, -4.5])
    padded = _wide(
        [True] * 4 + [False] * 20,
        [-1.0, -2.0, -3.0, -4.0] + [-1.0] * 20,
        [-1.5, -2.5, -2.5, -4.5] + [-1.0] * 20,
    )

    a = wf.paired_difference(lagged, "elpd_spoken", restriction="lagged")
    b = wf.paired_difference(padded, "elpd_spoken", restriction="lagged")
    c = wf.paired_difference(padded, "elpd_spoken", restriction="all-later-waves")

    assert a == b  # restricting recovers the lagged-only answer exactly
    assert c["elpd_diff"] == pytest.approx(a["elpd_diff"])
    assert c["se"] < a["se"]
    assert c["n_rows"] == 24 and a["n_rows"] == 4


def test_the_headline_restriction_drops_rows_whose_source_was_held_out():
    """``lagged-from-training`` is the strict subset of ``lagged``.

    The two answer different questions and the difference between them is not
    presentational: one is prediction from data the model was fitted to, the
    other conditions on a row it never saw.
    """
    wide = _wide(
        [True] * 4,
        [-1.0, -2.0, -3.0, -4.0],
        [-1.5, -2.5, -2.5, -4.5],
        clean=[True, True, False, False],
    )
    strict = wf.paired_difference(wide, "elpd_spoken", restriction="lagged-from-training")
    loose = wf.paired_difference(wide, "elpd_spoken", restriction="lagged")
    assert strict["n_rows"] == 2 and loose["n_rows"] == 4
    assert strict["elpd_diff"] == pytest.approx(1.0)


def test_every_named_restriction_is_understood():
    """The names are written into the output table, so a typo must not pass."""
    wide = _wide([True] * 3, [-1.0, -2.0, -3.0], [-1.5, -2.5, -2.5])
    for restriction in wf.RESTRICTIONS:
        out = wf.paired_difference(wide, "elpd_spoken", restriction=restriction)
        assert out["rows_scored"] == restriction
    with pytest.raises(ValueError):
        wf.paired_difference(wide, "elpd_spoken", restriction="everything")


def test_a_row_either_arm_could_not_score_is_dropped_from_both():
    wide = _wide([True] * 3, [-1.0, np.nan, -3.0], [-1.5, -2.5, -2.5])
    out = wf.paired_difference(wide, "elpd_spoken", restriction="lagged")
    assert out["n_rows"] == 2
    assert out["elpd_diff"] == pytest.approx((-1.0 + 1.5) + (-3.0 + 2.5))


def test_too_few_rows_reports_nan_rather_than_a_spurious_interval():
    wide = _wide([True], [-1.0], [-1.5])
    out = wf.paired_difference(wide, "elpd_spoken", restriction="lagged")
    assert out["n_rows"] == 1
    assert np.isnan(out["se"])
