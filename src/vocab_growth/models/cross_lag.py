# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""VG16's within-child cross-lag: the prior-wave understood source and its audit.

Pure NumPy/pandas, PyMC-free, in the manner of :mod:`observation_arrays` -- the
whole of what the cross-lag needs before a graph exists, plus the artefact that
records the coefficient's support. It was a quarter of ``common_bivariate_re.py``,
which fits eleven other models that carry no lag at all.

Read :func:`prev_wave_lag_for_frame` first: it is the supported entry point, and
:func:`prev_wave_lag` is the array primitive underneath it. The distinction matters
because the two settings that change the result -- the gap ceiling and the
zero-source treatment -- live on the definition, and a caller reaching past the
frame-level function has to pass them itself.

The one thing here that is *not* a definition-level concern is
:func:`validate_cross_lag`, which is checked against the resolved child-effect plan;
its docstring says why that check cannot move into
``definitions.validate_model_definition`` with the others.
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


def prev_wave_lag(
    subject,
    age,
    understood,
    n_trials,
    *,
    max_gap_months: float | None = None,
    zero_handling: str = LAG_ZERO_CLIP,
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
        with_u = wave[~np.isnan(understood[wave])]
        if with_u.size:
            source = int(with_u[np.argmax(understood[with_u])])
    # A gap ceiling drops the lag rather than the row: the observation still
    # enters both likelihoods, it simply stops informing `beta_lag`. Applied
    # after the source is chosen, so which wave is the source never depends on
    # the ceiling -- only whether that source is used.
    if max_gap_months is not None:
        too_far = (has_lag_f > 0) & ((age - age[prev_idx]) > max_gap_months)
        has_lag_f = np.where(too_far, 0.0, has_lag_f)
        prev_idx = np.where(too_far, 0, prev_idx)

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
    It reads the two settings that change the result off ``definition``, so a
    caller cannot silently get the registered defaults for a variant that moved
    them -- which is what ``definition=None`` used to allow, and what two of the
    three out-of-module callers were doing.

    ``definition`` is required for that reason: every caller has one. An array-only
    caller (a trace-reconstruction script) calls :func:`prev_wave_lag` directly and
    passes the same two settings itself.
    """
    return prev_wave_lag(
        np.asarray(analysis_df["subject_code"], dtype=int),
        np.asarray(analysis_df["age"], dtype=float),
        analysis_df["understood"].to_numpy(dtype=float),
        n_trials,
        max_gap_months=getattr(definition, "lag_max_gap_months", None),
        zero_handling=getattr(definition, "lag_zero_handling", LAG_ZERO_CLIP),
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
