# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The within-child cross-lags: their prior-wave sources and their audits.

Pure NumPy/pandas, PyMC-free, in the manner of :mod:`observation_arrays` -- the
whole of what the cross-lag needs before a graph exists, plus the artefact that
records the coefficient's support. It was a quarter of ``common_bivariate_re.py``,
which fits eleven other models that carry no lag at all.

Read :func:`prev_wave_lag_for_frame` first: it is the supported entry point, and
:func:`prev_wave_lag` is the array primitive underneath it. The distinction matters
because the three settings that change the result -- the gap ceiling, the
zero-source treatment and the same-form restriction -- live on the definition,
and a caller reaching past the frame-level function has to pass them itself.

The one thing here that is *not* a definition-level concern is
:func:`validate_cross_lag`, which is checked against the resolved child-effect plan;
its docstring says why that check cannot move into
``definitions.validate_model_definition`` with the others.

Two lags live here, and they share everything except what they read. VG16's
understood -> ``q`` lag reads a **count** against the fixed 810-item inventory
(:func:`prev_wave_lag`); VG25's sign -> speech lag reads a **ratio**, the signed
share of the same wave's comprehension (:func:`prev_wave_sign_share_lag`). The
wave walk, the source-selection rule, the gap ceiling and the zero treatment are
one implementation between them -- :func:`_assign_prev_wave_sources` and
:func:`_apply_gap_ceiling` -- so a correction to either lands on both.

One thing deliberately does *not* carry over. ``same_form_only`` exists for the
count lag because ``understood / 810`` is deflated by a shorter source form, and
a study intercept cannot absorb a within-study form transition. The ratio lag
divides one count by another **scored on the same form in the same
administration**, so the truncation is very largely common to numerator and
denominator and cancels. That is a property of the predictor, not an assumption
about the pool, so VG25 carries no same-form field rather than carrying one that
would always be inert.
"""

import os

import numpy as np
import pandas as pd

from vocab_growth.models.likelihood_utils import (
    LAG_BASELINES,
    LAG_ZERO_CLIP,
    LAG_ZERO_CONTINUITY,
    LAG_ZERO_TREATMENTS,
)
from vocab_growth.reporting import key_value_table


def validate_cross_lag(lag_baseline: str, subject_re_u_active: bool) -> None:
    """Validate the VG16 within-child cross-lag configuration (issue #113).

    ``lag_baseline`` must be one of :data:`LAG_BASELINES`, which is also what
    ``definitions.validate_model_definition`` checks the *field* against -- the two
    cannot drift because they read one tuple.

    Both baselines are defined relative to the child's understood subject
    intercept -- the within-child baseline subtracts it, the population-relative
    baseline adds it back -- so a comprehension child effect must be present;
    otherwise the two baselines silently coincide (and the population branch would
    index a scalar).

    ``subject_re_u_active`` is the **resolved** ``SubjectEffectPlan["u"].is_active``,
    not ``definition.use_subject_re_u``, and that is why this check is not simply
    folded into ``validate_model_definition``: the definition-level check can only
    see the raw field, while a plan can deactivate a declared effect. Both checks
    are worth having, and only this one sees what the graph will actually contain.
    """
    if lag_baseline not in LAG_BASELINES:
        raise ValueError(
            f"lag_baseline must be one of {LAG_BASELINES}, got {lag_baseline!r}."
        )
    if not subject_re_u_active:
        raise ValueError(
            "Cross-lag (use_cross_lag=True) requires use_subject_re_u=True: both the "
            "population-relative and within-child baselines are defined relative to "
            "the child's understood subject intercept."
        )


def iter_subject_age_waves(subject, age):
    """Yield each ``(subject, recorded age)`` administration wave as one group.

    A wave is every row a child carries at one recorded age, taken complete:
    the indices are yielded together so a caller can assign one prior-wave
    state to all of them before any of them advances that state. Children are
    walked in code order and each child's waves in increasing age. The set of
    indices in each yielded wave is invariant to the input row order; only
    their order inside the wave follows it.
    """
    order = np.lexsort((age, subject))
    n = len(order)
    start = 0
    while start < n:
        stop = start
        s, a = subject[order[start]], age[order[start]]
        while stop < n and subject[order[stop]] == s and age[order[stop]] == a:
            stop += 1
        yield order[start:stop]
        start = stop


def wave_index(subject, age):
    """0 for each child's first administration wave, 1 for the next, and so on.

    The per-row counterpart of :func:`iter_subject_age_waves`, built on it so
    every row at one recorded age takes the same index: a child measured on two
    forms on one day has one wave, not two, which is the wave definition issue
    #242 settled. ``kfold_loso.py`` and ``wave_forward_score.py`` both read it
    and each carried a verbatim copy until the 2026-09-09 refit window, because
    adding a function here moves the executable-code signature.
    """
    subject = np.asarray(subject)
    age = np.asarray(age, dtype=float)
    index = np.zeros(len(subject), dtype=int)
    current = None
    counter = 0
    for rows in iter_subject_age_waves(subject, age):
        s = subject[rows[0]]
        if current is None or s != current:
            current = s
            counter = 0
        index[rows] = counter
        counter += 1
    return index


def _assign_prev_wave_sources(subject, age, usable, rank):
    """Point every row at its child's most recent strictly earlier usable wave.

    The walk both lags share. ``usable`` marks the rows that can *serve* as a
    source; ``rank`` is the quantity maximised to choose between several usable
    measurements inside one source wave, and is the least-truncated-measurement
    rule in both cases -- the largest understood count for VG16's count lag, and
    the largest comprehension denominator for VG25's ratio lag, which is the
    same thing said of a ratio.

    Returns ``(prev_idx, has_lag_f)``: the selected source row (0 where absent,
    gated by ``has_lag_f``) and 1.0/0.0 for whether one exists at all. The state
    advances only once a whole wave is assigned, so no row can take a same-age
    source and the result does not depend on the input row order.
    """
    subject = np.asarray(subject, dtype=int)
    age = np.asarray(age, dtype=float)
    usable = np.asarray(usable, dtype=bool)
    rank = np.asarray(rank, dtype=float)
    n = len(subject)
    prev_idx = np.zeros(n, dtype=int)
    has_lag_f = np.zeros(n, dtype=float)
    current_subject, source = -1, -1
    for wave in iter_subject_age_waves(subject, age):
        s = subject[wave[0]]
        if s != current_subject:
            current_subject, source = s, -1
        if source >= 0:
            prev_idx[wave] = source
            has_lag_f[wave] = 1.0
        candidates = wave[usable[wave]]
        if candidates.size:
            source = int(candidates[np.argmax(rank[candidates])])
    return prev_idx, has_lag_f


def _apply_gap_ceiling(prev_idx, has_lag_f, age, max_gap_months):
    """Drop -- not the row -- every lag reaching further back than the ceiling.

    Applied after the source is chosen, so which wave is the source never
    depends on the ceiling; only whether that source is used. The observation
    still enters every likelihood it did before, it simply stops informing the
    coefficient.
    """
    if max_gap_months is None:
        return prev_idx, has_lag_f
    age = np.asarray(age, dtype=float)
    too_far = (has_lag_f > 0) & ((age - age[prev_idx]) > max_gap_months)
    return np.where(too_far, 0, prev_idx), np.where(too_far, 0.0, has_lag_f)


def prev_wave_lag(
    subject,
    age,
    understood,
    n_trials,
    *,
    max_gap_months: float | None = None,
    zero_handling: str = LAG_ZERO_CLIP,
    form_ceiling=None,
    same_form_only: bool = False,
):
    """Per-observation prior-wave understood lag source for the VG16 cross-lag.

    The unit is an **administration wave**: every row a child carries at one
    recorded age, processed as a complete group (issue #242).

    * Every row in a wave receives the same source — the child's most recent
      strictly earlier wave with at least one usable understood count,
      skipping earlier waves without one.
    * The source state advances only after a whole wave is assigned, so a row
      can never receive a same-age source and the result is invariant to the
      input row order. The row-by-row walk this replaced advanced state
      immediately after each row, so which of two same-recorded-age rows
      (two checklist forms) carried the lag depended on arbitrary tie order —
      66 spoken observations from 46 children lost their lag to it on the
      2026-08 frame.
    * Where a source wave carries several understood measurements (two forms
      at one recorded age), the largest count is selected: every count is
      scored against the same ``n_trials`` inventory under the project's
      difficulty-ordering harmonisation, and a shorter form right-truncates
      it, so the largest observed count is the least-truncated measurement
      available. On the current frame no source wave carries more than one
      understood measurement, so the rule is registered ahead of need. Rows
      of one wave share child, study and recorded age, so which *row* the
      source index points at cannot move the likelihood — only the selected
      count can.

    ``same_form_only`` (with ``form_ceiling``) keeps only lags whose source and
    target waves were scored against the same checklist, which is the review's
    own measurement check: the predictor is a logit of ``understood /
    n_trials``, so a source scored on a shorter form enters it already
    deflated, and a study intercept cannot absorb a *within*-study form
    transition. Like the gap ceiling it drops the lag, not the row.

    Returns ``(prev_idx, has_lag_f, y_u_prev_logit)`` as per-observation
    arrays: ``has_lag_f`` is 1.0 where a source wave exists and 0.0 otherwise
    (a child's first wave, or when every earlier wave lacks comprehension);
    ``prev_idx`` points at the selected source row (0 where absent, gated by
    ``has_lag_f``); ``y_u_prev_logit`` is the logit of the source understood
    proportion (clipped away from 0/1), and 0.0 where there is no lag source.
    """
    subject = np.asarray(subject, dtype=int)
    age = np.asarray(age, dtype=float)
    understood = np.asarray(understood, dtype=float)
    n = len(subject)
    prev_idx, has_lag_f = _assign_prev_wave_sources(
        subject, age, ~np.isnan(understood), understood
    )
    prev_idx, has_lag_f = _apply_gap_ceiling(prev_idx, has_lag_f, age, max_gap_months)
    # The same-form restriction drops the lag on the same terms and for the same
    # reason: applied after the source is chosen, so a row whose source used a
    # different form loses its lag rather than falling back to an earlier
    # same-form wave, which would silently lengthen the gap and confound the
    # measurement question with the interval one. Both ceilings must be known --
    # an unknown one cannot certify that the two waves used the same checklist.
    if same_form_only:
        if form_ceiling is None:
            raise ValueError(
                "same_form_only needs form_ceiling: the restriction is defined on "
                "the checklist each wave was scored against, and none was given."
            )
        ceiling = np.asarray(form_ceiling, dtype=float)
        if len(ceiling) != n:
            raise ValueError(
                f"form_ceiling has {len(ceiling)} entries for {n} observations."
            )
        source_ceiling = ceiling[prev_idx]
        same = (
            (ceiling == source_ceiling)
            & ~np.isnan(ceiling)
            & ~np.isnan(source_ceiling)
        )
        crossed = (has_lag_f > 0) & ~same
        has_lag_f = np.where(crossed, 0.0, has_lag_f)
        prev_idx = np.where(crossed, 0, prev_idx)

    und_prev = np.where(has_lag_f > 0, understood[prev_idx], n_trials * 0.5)
    if zero_handling == LAG_ZERO_CONTINUITY:
        p_prev = (und_prev + 0.5) / (n_trials + 1.0)
    elif zero_handling == LAG_ZERO_CLIP:
        p_prev = np.clip(und_prev / n_trials, 1e-4, 1 - 1e-4)
    else:
        raise ValueError(
            f"Unknown lag_zero_handling {zero_handling!r}; expected one of "
            + ", ".join(map(repr, LAG_ZERO_TREATMENTS))
        )
    y_u_prev_logit = np.where(has_lag_f > 0, np.log(p_prev) - np.log(1 - p_prev), 0.0)
    return prev_idx, has_lag_f, y_u_prev_logit


def prev_wave_lag_for_frame(analysis_df, n_trials: int, definition):
    """The supported entry point: :func:`prev_wave_lag` over an analysis frame.

    Call this, not :func:`prev_wave_lag`, wherever an analysis frame is in hand.
    It reads the three settings that change the result off ``definition``, so a
    caller cannot silently get the registered defaults for a variant that moved
    them -- which is what ``definition=None`` used to allow, and what two of the
    three out-of-module callers were doing.

    ``definition`` is required for that reason: every caller has one. An array-only
    caller (a trace-reconstruction script) calls :func:`prev_wave_lag` directly and
    passes the same three settings itself -- and, for the same-form restriction,
    the form identity this function reads off the frame.
    """
    same_form_only = bool(getattr(definition, "lag_same_form_only", False))
    form_ceiling = None
    if same_form_only:
        if "survey_vocab_max" not in analysis_df.columns:
            raise ValueError(
                "lag_same_form_only=True needs the frame's `survey_vocab_max` "
                "column, and this prepared frame does not carry it. The engine "
                "requests it for every Down syndrome frame and for `use_cross_lag` "
                "specifically, so the only way here is a typically-developing "
                "cross-lag model: the Wordbank query never produces the column, "
                "and form identity would have to come from `form` instead."
            )
        form_ceiling = analysis_df["survey_vocab_max"].to_numpy(dtype=float)
    return prev_wave_lag(
        np.asarray(analysis_df["subject_code"], dtype=int),
        np.asarray(analysis_df["age"], dtype=float),
        analysis_df["understood"].to_numpy(dtype=float),
        n_trials,
        max_gap_months=getattr(definition, "lag_max_gap_months", None),
        zero_handling=getattr(definition, "lag_zero_handling", LAG_ZERO_CLIP),
        form_ceiling=form_ceiling,
        same_form_only=same_form_only,
    )


# ============================================================
# VG25's sign -> speech lag: the prior-wave signed share of comprehension
# ============================================================

#: The two within-understood cross-tab cells that sum to a wave's signed total.
SIGN_SHARE_CELL_COLUMNS = ("signed_only", "signed_spoken")


def validate_sign_cross_lag(
    sign_lag_baseline: str, subject_re_sign_active: bool
) -> None:
    """Validate the VG25 sign -> speech cross-lag configuration (issue #297).

    The counterpart of :func:`validate_cross_lag`, and the same two checks
    against the same :data:`LAG_BASELINES` tuple -- but defined relative to the
    child's **signed-ratio** intercept rather than their understood one, because
    that is the trajectory the predictor is a residual from. The within-child
    baseline subtracts it and the population-relative baseline does not, so
    without a sign child effect the two silently coincide.

    Separate from :func:`validate_cross_lag` rather than parameterised: the two
    lags can be configured independently, and a message naming the wrong flag is
    worse than a second function.
    """
    if sign_lag_baseline not in LAG_BASELINES:
        raise ValueError(
            f"sign_lag_baseline must be one of {LAG_BASELINES}, got "
            f"{sign_lag_baseline!r}."
        )
    if not subject_re_sign_active:
        raise ValueError(
            "Sign cross-lag (use_sign_cross_lag=True) requires "
            "use_subject_re_sign=True: both the population-relative and "
            "within-child baselines are defined relative to the child's "
            "signed-ratio subject intercept."
        )


def sign_share_counts(analysis_df):
    """``(signed, understood)`` per row, NaN where no signed share is measurable.

    The signed share of comprehension has two sources in the joint frame, and
    one near-miss that is deliberately not a third.

    * **Marginal rows** carry ``signed`` against ``understood`` directly.
    * **Within-understood cross-tab rows** (uk_02, uk_07, es_01) carry ``signed``
      as NaN -- deliberately, so the cells are not double counted against the
      marginal -- but their cells partition the same comprehension total, so
      ``signed_only + signed_spoken`` over ``understood`` is the same quantity
      measured the same way. Without this branch the frame's cross-tab rows
      would be invisible to the lag, which is most of uk_02 and all of uk_07.
    * **nz_01's three cells partition PRODUCED words**, so the only share they
      support is the signed share of *production*. That is a different variable,
      not a differently-denominated version of this one, and pooling the two
      under one coefficient would make ``beta_sign_lag`` mean two things at once.
      Those rows supply no source; the decision is recorded on
      :class:`~vocab_growth.models.definitions.JointCrossLagModelDefinition`.

    A row needs a strictly positive comprehension denominator as well as a
    numerator: ``signed / 0`` is undefined, not zero, and a wave that understood
    nothing measures no share of it.
    """
    understood = analysis_df["understood"].to_numpy(dtype=float)
    signed = analysis_df["signed"].to_numpy(dtype=float)
    if all(column in analysis_df.columns for column in SIGN_SHARE_CELL_COLUMNS):
        from_cells = sum(
            analysis_df[column].to_numpy(dtype=float)
            for column in SIGN_SHARE_CELL_COLUMNS
        )
        signed = np.where(np.isnan(signed), from_cells, signed)
    usable = ~np.isnan(signed) & ~np.isnan(understood) & (understood > 0)
    return (
        np.where(usable, signed, np.nan),
        np.where(usable, understood, np.nan),
    )


def prev_wave_sign_share_lag(
    subject,
    age,
    signed,
    understood,
    *,
    max_gap_months: float | None = None,
    zero_handling: str = LAG_ZERO_CLIP,
):
    """Per-observation prior-wave signed-share lag source for the VG25 cross-lag.

    The same administration-wave unit, the same source walk and the same gap
    ceiling as :func:`prev_wave_lag` -- see there for what a wave is and why the
    source advances only once a whole wave is assigned. What differs is what the
    source measures: the **logit of the signed share of comprehension** at the
    prior wave, ``logit(signed / understood)``, rather than the logit of an
    understood count against the fixed inventory.

    ``signed`` and ``understood`` are the pair :func:`sign_share_counts` returns,
    NaN-aligned: a row is usable as a source exactly where both are present and
    the denominator is positive.

    Where a source wave carries several usable measurements, the one with the
    **largest comprehension denominator** is selected. That is the same
    least-truncated-measurement rule :func:`prev_wave_lag` applies to the count,
    said of a ratio: a shorter form right-truncates both counts, and the wave's
    largest denominator is its least-truncated view of the child. The ratio is
    also the reason no same-form restriction is offered -- see the module
    docstring.

    Returns ``(prev_idx, has_lag_f, r_prev_logit)`` as per-observation arrays,
    with the same meanings :func:`prev_wave_lag` documents. ``r_prev_logit`` is
    0.0 where there is no lag source.
    """
    subject = np.asarray(subject, dtype=int)
    age = np.asarray(age, dtype=float)
    signed = np.asarray(signed, dtype=float)
    understood = np.asarray(understood, dtype=float)
    usable = ~np.isnan(signed) & ~np.isnan(understood) & (understood > 0)
    prev_idx, has_lag_f = _assign_prev_wave_sources(
        subject, age, usable, np.where(usable, understood, -np.inf)
    )
    prev_idx, has_lag_f = _apply_gap_ceiling(prev_idx, has_lag_f, age, max_gap_months)

    # A neutral 0.5 / 1.0 placeholder where there is no source: both treatments
    # map it to logit(0.5) = 0, which `has_lag_f` then zeroes anyway. Written so
    # neither branch can divide by a NaN.
    signed_prev = np.where(has_lag_f > 0, signed[prev_idx], 0.5)
    understood_prev = np.where(has_lag_f > 0, understood[prev_idx], 1.0)
    # A numerator above its own denominator is a share above 1, which the clip
    # absorbs and the continuity correction does NOT: (k + 0.5) / (n + 1) stays
    # above 1, and `log(1 - r)` of it is a silent NaN that would propagate into
    # the log density. It cannot happen on the frames registered today -- the
    # loader masks a comprehension count that falls below the child's recorded
    # production union, and a cross-tab's cells sum to its own total by
    # construction, so `signed <= understood` on all 562 rows carrying a share.
    # It becomes reachable the moment that mask is reinstated for a sensitivity,
    # which is one field away. Clipped here rather than guarded at the call site
    # so both treatments see a well-defined share, and a no-op on valid data.
    signed_prev = np.clip(signed_prev, 0.0, understood_prev)
    if zero_handling == LAG_ZERO_CONTINUITY:
        r_prev = (signed_prev + 0.5) / (understood_prev + 1.0)
    elif zero_handling == LAG_ZERO_CLIP:
        r_prev = np.clip(signed_prev / understood_prev, 1e-4, 1 - 1e-4)
    else:
        raise ValueError(
            f"Unknown sign_lag_zero_handling {zero_handling!r}; expected one of "
            + ", ".join(map(repr, LAG_ZERO_TREATMENTS))
        )
    # The clip bites at BOTH ends here and only at the lower end for the count
    # lag, because a signed share of exactly 1 is reachable -- a child who signs
    # every word they understand -- while an understood count of n_trials is not
    # in this pool. Both boundaries are data rather than defects, which is what
    # the continuity treatment exists to say differently.
    r_prev_logit = np.where(has_lag_f > 0, np.log(r_prev) - np.log(1 - r_prev), 0.0)
    return prev_idx, has_lag_f, r_prev_logit


def prev_wave_sign_share_lag_for_frame(analysis_df, definition):
    """The supported entry point: :func:`prev_wave_sign_share_lag` over a frame.

    Call this, not the primitive, wherever an analysis frame is in hand, for the
    reason :func:`prev_wave_lag_for_frame` gives: the two settings that change
    the result live on the definition, and a caller reaching past this function
    has to pass them itself.

    ``n_trials`` is absent by construction. The predictor is a ratio of two
    counts from one administration, so the inventory size cancels out of it --
    which is also why this lag has no same-form restriction to configure.
    """
    signed, understood = sign_share_counts(analysis_df)
    return prev_wave_sign_share_lag(
        np.asarray(analysis_df["subject_code"], dtype=int),
        np.asarray(analysis_df["age"], dtype=float),
        signed,
        understood,
        max_gap_months=getattr(definition, "sign_lag_max_gap_months", None),
        zero_handling=getattr(definition, "sign_lag_zero_handling", LAG_ZERO_CLIP),
    )


def cross_lag_audit_frame(
    analysis_df,
    prev_idx,
    has_lag_f,
    spoken_indices,
    spoken_is_conditional,
):
    """One row per observation with a prior-wave understood source (issue #242).

    Persists the cross-lag coefficient's support as a fit artefact so reports
    read the counts from a file instead of restating them: the source wave and
    its gap, the selected source count (flagging clipped zeros and waves where
    the largest-count selection had more than one measurement to choose from),
    whether the row enters the spoken likelihood and on which branch, and —
    where the frame carries form ceilings — the checklist transition between
    the source and current waves.
    """
    n = len(analysis_df)
    branch = np.full(n, "", dtype=object)
    branch[np.asarray(spoken_indices, dtype=int)] = np.where(
        np.asarray(spoken_is_conditional, dtype=bool), "conditional", "marginal"
    )
    lagged = np.flatnonzero(np.asarray(has_lag_f, dtype=float) > 0)
    src = np.asarray(prev_idx, dtype=int)[lagged]
    subj = np.asarray(analysis_df["subject_code"], dtype=int)
    age = np.asarray(analysis_df["age"], dtype=float)
    und = analysis_df["understood"].to_numpy(dtype=float)
    # Understood measurements available at each child-age wave, keyed so the
    # audit can say how often the largest-count source selection actually had
    # a choice to make.
    wave_u_counts = (
        analysis_df.assign(_subj=subj, _age=age)
        .groupby(["_subj", "_age"])["understood"]
        .count()
    )
    src_keys = list(zip(subj[src], age[src], strict=True))
    frame = pd.DataFrame(
        {
            "row": lagged,
            "subject_code": subj[lagged],
            "age_months": age[lagged],
            "source_row": src,
            "source_age_months": age[src],
            "gap_months": age[lagged] - age[src],
            "source_understood": und[src],
            "source_understood_zero": und[src] == 0,
            "source_wave_understood_measurements": [
                int(wave_u_counts.loc[k]) for k in src_keys
            ],
            # "" = the row carries no spoken observation in the likelihood, so
            # its lag cannot inform beta_lag.
            "spoken_branch": branch[lagged],
        }
    )
    if "study" in analysis_df.columns:
        frame.insert(2, "study", np.asarray(analysis_df["study"])[lagged])
    if "survey_vocab_max" in analysis_df.columns:
        ceilings = analysis_df["survey_vocab_max"].to_numpy(dtype=float)
        frame["source_form_ceiling"] = ceilings[src]
        frame["form_ceiling"] = ceilings[lagged]
        frame["form_ceiling_changed"] = (
            (ceilings[lagged] != ceilings[src])
            & ~np.isnan(ceilings[lagged])
            & ~np.isnan(ceilings[src])
        )
    return frame


def report_cross_lag_support(
    output_dir: str, audit: pd.DataFrame, n_obs: int
) -> None:
    """Write ``cross_lag_audit.csv`` and print the support summary (issue #242).

    Takes the directory rather than the fit context: it is the only write in this
    module, and passing a whole ``ModelFitContext`` for one path was what kept the
    block in the engine.
    """
    audit.to_csv(os.path.join(output_dir, "cross_lag_audit.csv"), index=False)
    supporting = audit[audit["spoken_branch"] != ""]
    gaps = supporting["gap_months"]
    rows: list[tuple[str, object]] = [
        ("Observations with a prior-wave understood source", len(audit)),
        ("... of them entering the spoken likelihood", len(supporting)),
        ("Children contributing a supporting observation", supporting["subject_code"].nunique()),
        ("Supporting rows on the conditional S|U branch", int((supporting["spoken_branch"] == "conditional").sum())),
        ("Supporting rows on the marginal fallback branch", int((supporting["spoken_branch"] == "marginal").sum())),
        (
            "Gap to source (months): median (IQR) [range]",
            f"{gaps.median():.1f} ({gaps.quantile(0.25):.1f}-{gaps.quantile(0.75):.1f}) "
            f"[{gaps.min():.0f}-{gaps.max():.0f}]"
            if len(supporting)
            else "n/a",
        ),
        ("Zero-count sources (clipped logit)", int(supporting["source_understood_zero"].sum())),
        (
            "Source waves offering >1 understood measurement",
            int((supporting["source_wave_understood_measurements"] > 1).sum()),
        ),
    ]
    if "form_ceiling_changed" in supporting.columns:
        rows.append(
            (
                "Supporting rows changing form ceiling source -> target",
                int(supporting["form_ceiling_changed"].sum()),
            )
        )
    rows.append(("Observations in the frame", n_obs))
    key_value_table("Cross-lag support (cross_lag_audit.csv)", rows)


def sign_cross_lag_audit_frame(
    analysis_df,
    prev_idx,
    has_lag_f,
    *,
    spoken_indices,
    spoken_is_conditional,
    cell_indices,
    prod_indices,
):
    """One row per observation with a prior-wave signed-share source (#297).

    The counterpart of :func:`cross_lag_audit_frame`, and it exists for the same
    reason: the coefficient's support is persisted as a fit artefact so reports
    read the counts from a file rather than restating them.

    What differs is the ``branch`` column, because the joint engine has four
    places a lagged row can land rather than two. ``conditional`` and
    ``marginal`` are the spoken marginal likelihood's two branches, exactly as in
    the bivariate audit; ``cells`` is a within-understood four-cell row and
    ``produced-cells`` an nz_01 three-cell row. The four are disjoint by
    construction -- a cross-tab row carries no spoken marginal -- and an empty
    string means the row informs ``beta_sign_lag`` through nothing at all, which
    is what makes the file an audit rather than a listing.

    ``source_signed_share`` is the predictor's raw input before any baseline is
    subtracted, so a reader can see the range the coefficient is identified over
    without rebuilding the graph; the boundary flags say how many rows sit where
    the zero treatment decides the value.
    """
    n = len(analysis_df)
    branch = np.full(n, "", dtype=object)
    branch[np.asarray(cell_indices, dtype=int)] = "cells"
    branch[np.asarray(prod_indices, dtype=int)] = "produced-cells"
    branch[np.asarray(spoken_indices, dtype=int)] = np.where(
        np.asarray(spoken_is_conditional, dtype=bool), "conditional", "marginal"
    )
    signed, understood = sign_share_counts(analysis_df)
    lagged = np.flatnonzero(np.asarray(has_lag_f, dtype=float) > 0)
    src = np.asarray(prev_idx, dtype=int)[lagged]
    subj = np.asarray(analysis_df["subject_code"], dtype=int)
    age = np.asarray(analysis_df["age"], dtype=float)
    share = signed[src] / understood[src]
    # How often the largest-denominator selection actually had a choice to make,
    # counted over the measurements that could have served -- not over every row
    # of the wave, most of which carry no signed share at all.
    usable = ~np.isnan(signed)
    wave_sources = (
        pd.DataFrame({"_subj": subj, "_age": age, "_usable": usable})
        .groupby(["_subj", "_age"])["_usable"]
        .sum()
    )
    frame = pd.DataFrame(
        {
            "row": lagged,
            "subject_code": subj[lagged],
            "age_months": age[lagged],
            "source_row": src,
            "source_age_months": age[src],
            "gap_months": age[lagged] - age[src],
            "source_signed": signed[src],
            "source_understood": understood[src],
            "source_signed_share": share,
            "source_share_zero": share == 0.0,
            "source_share_one": share == 1.0,
            "source_wave_share_measurements": [
                int(wave_sources.loc[key])
                for key in zip(subj[src], age[src], strict=True)
            ],
            # "" = the row enters no likelihood the lag term reaches, so its lag
            # cannot inform beta_sign_lag.
            "branch": branch[lagged],
        }
    )
    if "study" in analysis_df.columns:
        frame.insert(2, "study", np.asarray(analysis_df["study"])[lagged])
    return frame


def report_sign_cross_lag_support(
    output_dir: str, audit: pd.DataFrame, n_obs: int, *, in_cells: bool
) -> None:
    """Write ``sign_cross_lag_audit.csv`` and print the support summary (#297).

    ``in_cells`` is the definition's own switch, not something inferred from the
    audit: it decides which branches the lag term actually reaches, and a
    summary that counted the cross-tab rows as support while the graph ignored
    them would overstate the coefficient's evidence by half.
    """
    audit.to_csv(os.path.join(output_dir, "sign_cross_lag_audit.csv"), index=False)
    reached = (
        {"conditional", "marginal", "cells", "produced-cells"}
        if in_cells
        else {"conditional", "marginal"}
    )
    supporting = audit[audit["branch"].isin(reached)]
    gaps = supporting["gap_months"]
    rows: list[tuple[str, object]] = [
        ("Observations with a prior-wave signed-share source", len(audit)),
        ("... of them reached by the lag term", len(supporting)),
        ("Children contributing a supporting observation", supporting["subject_code"].nunique()),
        ("Supporting rows on the conditional S|U branch", int((supporting["branch"] == "conditional").sum())),
        ("Supporting rows on the marginal fallback branch", int((supporting["branch"] == "marginal").sum())),
        ("Supporting rows in the four-cell composition", int((supporting["branch"] == "cells").sum())),
        ("Supporting rows in the produced-cell composition", int((supporting["branch"] == "produced-cells").sum())),
        (
            "Gap to source (months): median (IQR) [range]",
            f"{gaps.median():.1f} ({gaps.quantile(0.25):.1f}-{gaps.quantile(0.75):.1f}) "
            f"[{gaps.min():.0f}-{gaps.max():.0f}]"
            if len(supporting)
            else "n/a",
        ),
        (
            "Source signed share: median [range]",
            f"{supporting['source_signed_share'].median():.3f} "
            f"[{supporting['source_signed_share'].min():.3f}-"
            f"{supporting['source_signed_share'].max():.3f}]"
            if len(supporting)
            else "n/a",
        ),
        ("Sources at a share of exactly 0 (boundary)", int(supporting["source_share_zero"].sum())),
        ("Sources at a share of exactly 1 (boundary)", int(supporting["source_share_one"].sum())),
        (
            "Source waves offering >1 signed-share measurement",
            int((supporting["source_wave_share_measurements"] > 1).sum()),
        ),
        ("Observations in the frame", n_obs),
    ]
    key_value_table("Sign cross-lag support (sign_cross_lag_audit.csv)", rows)
