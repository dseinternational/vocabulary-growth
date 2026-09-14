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


def test_the_sign_lag_control_removes_the_sign_lag_and_nothing_else():
    """VG25's coefficient lives on a different field from VG16's.

    Its control is structurally VG24, but it is built by flipping VG25's own
    boolean rather than by naming VG24: the two definitions differ in more than
    the coefficient, and a comparison against VG24 would carry those differences
    into the elpd.
    """
    import dataclasses

    definition = MODEL_REGISTRY["vg25"]
    control = wf.control_definition(definition)
    changed = {
        f.name
        for f in dataclasses.fields(definition)
        if getattr(definition, f.name) != getattr(control, f.name)
    }
    assert changed == {"use_sign_cross_lag", "config_name"}
    assert control.use_sign_cross_lag is False


def test_only_cross_lag_models_can_be_scored():
    """Without a coefficient to remove, the two arms are the same model."""
    assert "vg16" in wf.CROSS_LAG_MODELS
    assert "vg25" in wf.CROSS_LAG_MODELS
    # VG24 is VG25 without the lag and VG10 is VG16 without it: both are the
    # controls, not subjects.
    assert "vg10" not in wf.CROSS_LAG_MODELS
    assert "vg24" not in wf.CROSS_LAG_MODELS
    for key in wf.CROSS_LAG_MODELS:
        assert wf.lag_field(MODEL_REGISTRY[key]) in wf.LAG_FIELDS


def test_a_model_with_no_lag_field_is_refused_by_name():
    """The message has to say what is missing, not just fail."""
    with pytest.raises(ValueError, match="cross-lag field"):
        wf.lag_field(MODEL_REGISTRY["vg24"])


def test_every_scorable_model_has_row_scoring_for_its_engine():
    """Fail-closed: a lag on a new engine needs its likelihood evaluated.

    `CROSS_LAG_MODELS` is derived from the registry, so registering a cross-lag
    puts a model in the ``--model`` choices whether or not this script can score
    it. Without this, that model would reach `row_elpds` and raise there, one
    fold-fit at a time.
    """
    from vocab_growth.models.catalogue import engine_for

    for key in wf.CROSS_LAG_MODELS:
        engine = engine_for(key).name
        assert engine in wf.OUTCOME_COLUMNS, (
            f"{key} runs on the {engine!r} engine, which has no entry in "
            "OUTCOME_COLUMNS and no row scoring."
        )


def test_the_spoken_difference_is_the_headline_on_every_engine():
    """The driver reports `comparison.iloc[0]`, so the order is the claim."""
    for columns in wf.OUTCOME_COLUMNS.values():
        assert columns[0] == "elpd_spoken"


def test_the_joint_engine_scores_the_composition_the_lag_enters():
    """VG25's scope decision put the lag in the cross-tab cells.

    Scoring the marginals alone would score the coefficient on the evidence that
    decision chose against -- the registration's own measurement was 191
    supporting observations with the cells against 111 without them.
    """
    assert "elpd_cells" in wf.OUTCOME_COLUMNS["joint"]
    assert "elpd_cells" not in wf.OUTCOME_COLUMNS["bivariate_re"]


def test_the_sign_lag_source_uses_the_ratio_rule_not_the_count_rule():
    """The two lags read different quantities and select sources differently.

    VG16 ranks candidate source waves on the comprehension count; VG25 reads a
    ratio and passes both of its quantities as selection keys. Taking the wrong
    rule would silently score a predictor the model never used.
    """
    from vocab_growth.models.cross_lag import prev_wave_sign_share_lag_for_frame

    frame = _frame([
        (1, 12.0, 100, 10),
        (1, 18.0, 200, 40),
        (2, 12.0, 80, 5),
        (2, 20.0, 150, 30),
    ])
    frame["signed"] = [20, 30, 10, 25]
    definition = MODEL_REGISTRY["vg25"]

    lagged, source = wf.lag_source(frame, definition)
    prev_idx, has_lag_f, _logit = prev_wave_sign_share_lag_for_frame(
        frame, definition
    )
    assert np.array_equal(lagged, np.asarray(has_lag_f, dtype=float) > 0)
    assert np.array_equal(source, np.asarray(prev_idx, dtype=int))


# --- The composition density --------------------------------------------------


def test_the_composition_density_matches_pymcs_own():
    """The Dirichlet-Multinomial is written out here, so it is checked there.

    The engine's likelihood is a `pm.DirichletMultinomial` on exactly these
    parameters; a hand-written density that disagrees with it would score every
    held-out composition against a distribution the model does not hold.
    """
    import numpy as np
    import pymc as pm

    counts = np.array([3.0, 5.0, 2.0, 7.0])
    total = counts.sum()
    alpha = np.array([0.9, 2.5, 1.4, 3.1])

    expected = float(
        pm.logp(
            pm.DirichletMultinomial.dist(n=int(total), a=alpha), counts.astype(int)
        ).eval()
    )
    # One chain, one draw: the log mean over a single draw is that draw's value.
    got = wf._dirichlet_multinomial_elpd(
        counts, total, alpha[None, None, :], np.log(1.0)
    )
    assert got == pytest.approx(expected, rel=1e-10)


def test_the_produced_composition_keeps_its_parameters_unrenormalised():
    """Dropping the "neither" cell and renormalising would be a different model.

    The engine's comment says so explicitly: the produced-cell concentration
    sums to `conc * P(produced | understood)` rather than to `conc`.
    """
    import numpy as np

    pi = np.array([[[0.4, 0.2, 0.3, 0.1]]])
    conc = np.array([[10.0]])
    alpha_prod = conc[:, :, None] * pi[:, :, 1:]
    assert alpha_prod.sum() == pytest.approx(10.0 * 0.6)
    assert alpha_prod.sum() != pytest.approx(10.0)


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


# --- Which rows the joint scorer scores ---------------------------------------


def _joint_frame() -> pd.DataFrame:
    """Two children, two waves each: one cross-tab study, one marginal study.

    Rows 1 and 3 are the later waves -- the ones a fold scores. Row 1 is a
    four-cell row and row 3 is not, which is the distinction the scorer has to
    make.
    """
    frame = pd.DataFrame(
        {
            "subject_code": [1, 1, 2, 2],
            "age": [12.0, 18.0, 12.0, 18.0],
            "understood": [100.0, 200.0, 100.0, 200.0],
            "spoken": [10.0, 40.0, 10.0, 40.0],
            "signed": [20.0, 30.0, 20.0, 30.0],
        }
    )
    frame["study_code"] = [0, 0, 1, 1]
    frame["study"] = ["cells", "cells", "marginal", "marginal"]
    # Child 1 is a cross-tab child; child 2 is not, so its cells are missing.
    frame["understood_only"] = [60.0, 140.0, np.nan, np.nan]
    frame["signed_only"] = [20.0, 20.0, np.nan, np.nan]
    frame["spoken_only"] = [10.0, 30.0, np.nan, np.nan]
    frame["signed_spoken"] = [10.0, 10.0, np.nan, np.nan]
    frame["cell_total"] = [100.0, 200.0, np.nan, np.nan]
    for column in ("prod_signed_only", "prod_spoken_only", "prod_signed_spoken"):
        frame[column] = np.nan
    frame["prod_total"] = np.nan
    return frame


def _joint_trace(n_obs: int) -> object:
    """One chain, one draw, so a log mean density is that draw's log density."""
    import types

    def _var(value, extra=()):
        return types.SimpleNamespace(
            values=np.full((1, 1, n_obs, *extra), float(value))
            if extra
            else np.full((1, 1, n_obs), float(value))
        )

    posterior = {
        "p_u_obs": _var(0.5),
        "q_obs": _var(0.4),
        "r_obs": _var(0.3),
        "kappa_u_obs": _var(60.0),
        "kappa_s_obs": _var(50.0),
        "kappa_sign_obs": _var(40.0),
        "conc": types.SimpleNamespace(values=np.full((1, 1), 20.0)),
    }
    pi = np.zeros((1, 1, n_obs, 4))
    pi[..., :] = np.array([0.45, 0.2, 0.25, 0.1])
    posterior["pi_cells_obs"] = types.SimpleNamespace(values=pi)
    return types.SimpleNamespace(posterior=posterior)


def _score_joint(frame):
    rows = np.array([1, 3])
    lagged = np.array([False, True, False, True])
    clean = lagged.copy()
    return wf._joint_row_elpds(
        frame,
        _joint_trace(len(frame)),
        rows,
        MODEL_REGISTRY["vg25"],
        lagged,
        clean,
    ).set_index("row")


def test_a_cross_tab_row_is_scored_on_its_composition_not_its_marginals():
    """The engine gives a four-cell row no spoken or signed marginal at all.

    `marginal_outcome_eligible` excludes it, because its production information
    is in the composition. Scoring a marginal density there would score a
    density the model does not hold.
    """
    scored = _score_joint(_joint_frame())
    assert np.isfinite(scored.loc[1, "elpd_cells"])
    assert np.isnan(scored.loc[1, "elpd_spoken"])
    assert np.isnan(scored.loc[1, "elpd_signed"])
    # Comprehension is not excluded: a cross-tab row still carries `y_u_obs`.
    assert np.isfinite(scored.loc[1, "elpd_understood"])


def test_a_marginal_row_is_scored_on_its_marginals_and_has_no_composition():
    scored = _score_joint(_joint_frame())
    assert np.isfinite(scored.loc[3, "elpd_spoken"])
    assert np.isfinite(scored.loc[3, "elpd_signed"])
    assert np.isfinite(scored.loc[3, "elpd_understood"])
    assert np.isnan(scored.loc[3, "elpd_cells"])


def test_the_composition_is_found_from_the_frame_not_the_training_mask():
    """The defect the first end-to-end run hit, pinned.

    ``obs_cells_mask`` marks the cross-tab rows **in the likelihood**, and a
    fold's held-out rows are excluded from it by construction -- so every row
    this scores is absent from that mask and reading it scored no composition at
    all. The run got as far as pivoting an all-missing column away and then
    failed on its absence, several minutes of fold fitting later.
    """
    frame = _joint_frame()
    scored = _score_joint(frame)
    assert scored["elpd_cells"].notna().sum() == 1

    # And the criterion really is the frame's own column: blank it and the row
    # becomes a marginal one.
    without = frame.copy()
    without.loc[1, "signed_spoken"] = np.nan
    rescored = _score_joint(without)
    assert np.isnan(rescored.loc[1, "elpd_cells"])
    assert np.isfinite(rescored.loc[1, "elpd_spoken"])


def test_the_composition_density_is_the_one_the_likelihood_holds():
    """Against the parameters the engine builds, computed independently here."""
    import pymc as pm

    scored = _score_joint(_joint_frame())
    counts = np.array([140.0, 20.0, 30.0, 10.0])
    alpha = 20.0 * np.array([0.45, 0.2, 0.25, 0.1])
    expected = float(
        pm.logp(pm.DirichletMultinomial.dist(n=200, a=alpha), counts.astype(int)).eval()
    )
    assert scored.loc[1, "elpd_cells"] == pytest.approx(expected, rel=1e-10)


# --- The wide table -----------------------------------------------------------


def _long(**outcomes) -> pd.DataFrame:
    """Two scored rows under two arms, with whichever outcome columns are given."""
    base = pd.DataFrame(
        {
            "fold": [0, 0, 0, 0],
            "row": [1, 3, 1, 3],
            "subject_code": [1, 2, 1, 2],
            "age_months": [18.0, 18.0, 18.0, 18.0],
            "has_lag": [True, True, True, True],
            "source_in_training": [True, True, True, True],
            # Two distinct values, which is what makes the cartesian expansion
            # visible: 2 rows x 2 subjects x 2 branches is 8.
            "spoken_branch": ["", "conditional", "", "conditional"],
            "arm": ["lag", "lag", "control", "control"],
        }
    )
    for name, values in outcomes.items():
        base[name] = values
    return base


def test_the_wide_table_has_one_row_per_scored_row():
    """`dropna=False` fabricated six rows out of two, and was tried first.

    A cross-tab row scores a composition and no spoken marginal, so the frame
    genuinely holds missing values in every run; the fix for a vanished column
    must not invent rows to keep one.
    """
    long = _long(
        elpd_spoken=[np.nan, -3.0, np.nan, -3.1],
        elpd_cells=[-9.0, np.nan, -9.1, np.nan],
        elpd_understood=[-1.0, -2.0, -1.1, -2.1],
        elpd_signed=[np.nan, -5.0, np.nan, -5.1],
    )
    wide = wf.wide_table(long, wf.OUTCOME_COLUMNS["joint"], ("lag", "control"))
    assert len(wide) == 2
    assert sorted(wide["row"]) == [1, 3]
    # The cross-tab row keeps its composition and has no spoken density.
    cross_tab = wide.set_index("row").loc[1]
    assert cross_tab["elpd_cells_lag"] == -9.0
    assert np.isnan(cross_tab["elpd_spoken_lag"])


def test_an_outcome_that_scored_nothing_is_an_empty_column_not_an_absent_one():
    """The first end-to-end run died here, fifteen minutes in.

    `pivot_table` drops a value column that is missing everywhere, and
    `paired_difference` then raises a `KeyError` on a name the run was told to
    report. An outcome with nothing to score is a real outcome with no rows.
    """
    long = _long(
        elpd_spoken=[-1.0, -3.0, -1.1, -3.1],
        elpd_cells=[np.nan] * 4,
        elpd_understood=[-1.0, -2.0, -1.1, -2.1],
        elpd_signed=[-4.0, -5.0, -4.1, -5.1],
    )
    wide = wf.wide_table(long, wf.OUTCOME_COLUMNS["joint"], ("lag", "control"))
    assert "elpd_cells_lag" in wide.columns
    assert wide["elpd_cells_lag"].isna().all()

    summary = wf.paired_difference(wide, "elpd_cells", restriction="lagged")
    assert summary["n_rows"] == 0
    assert np.isnan(summary["elpd_diff"])


def test_the_marginal_branch_uses_the_product_of_the_two_probabilities():
    """A row with no usable comprehension count is scored over the inventory.

    The engine's `nested_outcome_alpha_beta` switches to `p_u * q` on
    `n_trials` trials there, where the conditional branch uses `q` on the
    child's own understood count. Both branches exist in every joint frame, and
    only the conditional one is exercised by the frame above.
    """
    from scipy.stats import betabinom

    frame = _joint_frame()
    # Child 2's later wave loses its comprehension count, which is what sends
    # the outcome to the marginal branch.
    frame.loc[3, "understood"] = np.nan
    scored = _score_joint(frame)

    n_trials = MODEL_REGISTRY["vg25"].n_trials
    p = 0.5 * 0.4  # p_u_obs * q_obs, the fabricated trace's constants
    k = 50.0  # kappa_s_obs
    expected = float(betabinom.logpmf(40, n_trials, p * k, (1 - p) * k))
    assert scored.loc[3, "elpd_spoken"] == pytest.approx(expected, rel=1e-12)
    assert np.isnan(scored.loc[3, "elpd_understood"])
