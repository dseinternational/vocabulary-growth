# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Where VG16's outcomes are missing, and what that does to the cross-lag.

The question, asked 2026-09-07 for
[#242](https://github.com/dseinternational/vocabulary-growth/issues/242)'s
available-case item. The VG16 review found that the lag is active only where an
earlier understood count and a later spoken count are both observed, that
understood missingness is "partly structural because forms measure different
outcomes", and that VG16 "conditions on these available observations and has no
model for outcome-observation or form-assignment processes". None of that had
been counted, and the words *structural* and *partly* were doing the work.

Method, in four parts, all on the model's own prepared frame:

1. **Classify the missingness.** A checklist form is comprehension-measuring if
   it records an understood count *anywhere* in the pool -- a property of the
   form, which is what "structural" has to mean if it is to mean anything. The
   same test is applied to spoken by form and by study.
2. **Account for every row without a lag**, separating a child's first wave (no
   earlier wave exists at all) from a later row whose every earlier wave lacked
   comprehension, and every child who can never contribute.
3. **Probe ignorability.** Under an available-case likelihood the observation
   process has to be ignorable given what the model carries -- study, age and
   the child effects. So: within a study and an age band, does the
   comprehension-less form go to the children with larger spoken vocabularies?
   Two confounds are handled rather than assumed away. Waves carrying *both*
   form classes are dropped from the between-child comparison, because the
   longer form's larger item pool would produce an association on its own; that
   artefact is measured separately on exactly those waves. And the band
   comparison is repeated at *exactly matched* recorded ages, which is the only
   version with no residual age gradient inside it.
4. **Support under each registered restriction**, rebuilding each variant's own
   frame through its own definition -- a data restriction changes which waves
   exist and therefore changes the lag, which subsetting the baseline's rows
   would not show.

Result: recorded in ``notes/202609071000-vg16-available-case-audit.md``. The
missingness is overwhelmingly structural (391 of 448 understood-missing rows are
on forms that never measure comprehension; 284 of 287 spoken-missing rows are
``us_03``, which records no production at all), which is the benign half. The
other half is that form assignment is not a function of age alone: at a fixed
study and age band the children on the comprehension-less form have
substantially larger spoken vocabularies, and the parallel-form item-pool
artefact is far too small to explain it. That is what motivated registering
``("vg16", "lag-same-form")``.

Reads the prepared DuckDB only -- no trace, no fit, nothing written.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from vocab_growth.analysis_frames import build_analysis_frame
from vocab_growth.models.cross_lag import prev_wave_lag_for_frame
from vocab_growth.models.definitions import MODEL_REGISTRY
from vocab_growth.models.observation_arrays import prepare_bivariate_observations
from vocab_growth.sensitivity.registry import build_variant, variants_for

# The frame the fit of record was made on (its manifest records both), so a
# loader-rule change shows up here as a failed assertion rather than as a table
# that is quietly about a different dataset.
EXPECTED_ROWS = 1_708
EXPECTED_CHILDREN = 943
N_TRIALS = 810
AGE_BANDS = [0, 24, 30, 36, 48, 60, 200]


def _frame(definition):
    frame, _ = build_analysis_frame("vg16", definition)
    return frame


def _lag(frame, definition):
    """The lag arrays and the rows that actually enter the spoken likelihood.

    The spoken rows come from the engine's own observation builder rather than
    from ``spoken.notna()``: under the paired-only treatment the marginal
    fallback rows are dropped from the likelihood, and a row that is not in it
    cannot inform the coefficient however good its lag.
    """
    prev_idx, has_lag_f, _ = prev_wave_lag_for_frame(frame, N_TRIALS, definition)
    observations = prepare_bivariate_observations(
        frame, definition, n_trials=N_TRIALS, use_subject_codes=True
    )
    in_spoken = np.zeros(len(frame), dtype=bool)
    in_spoken[observations.spoken_spec.indices] = True
    conditional = np.zeros(len(frame), dtype=bool)
    conditional[observations.spoken_spec.indices] = observations.spoken_spec.is_conditional
    return prev_idx, has_lag_f > 0, in_spoken, conditional


def _measures(frame: pd.DataFrame, column: str, by: str) -> pd.Series:
    """Whether each row's form (or study) records ``column`` anywhere in the pool."""
    ever = frame.groupby(by)[column].apply(lambda s: s.notna().any())
    return frame[by].map(ever)


def classify_missingness(frame: pd.DataFrame) -> None:
    print("== 1. Where the outcomes are missing ==\n")
    u_obs = frame["understood"].notna()
    s_obs = frame["spoken"].notna()
    print(f"rows {len(frame)}, children {frame['subject_code'].nunique()}, "
          f"studies {frame['study'].nunique()}")
    print(f"understood observed {int(u_obs.sum())}, missing {int((~u_obs).sum())}")
    print(f"spoken     observed {int(s_obs.sum())}, missing {int((~s_obs).sum())}")
    print(f"rows carrying both  {int((u_obs & s_obs).sum())}, "
          f"neither {int((~u_obs & ~s_obs).sum())}\n")

    form_measures_u = _measures(frame, "understood", "survey_vocab_max")
    never_u = sorted(
        frame.groupby("survey_vocab_max")["understood"]
        .apply(lambda s: s.notna().any())
        .loc[lambda s: ~s]
        .index
    )
    structural_u = ~u_obs & ~form_measures_u
    other_u = ~u_obs & form_measures_u
    print(f"understood missing, STRUCTURAL (form never measures it): "
          f"{int(structural_u.sum())}")
    print(f"    forms that never record an understood count: {never_u}")
    print(f"understood missing, OTHER (the form does measure it): "
          f"{int(other_u.sum())}")
    print(
        frame.loc[other_u]
        .groupby(["study", "survey_vocab_max"])
        .size()
        .rename("rows")
        .to_frame()
        .to_string()
    )
    print()

    study_measures_s = _measures(frame, "spoken", "study")
    never_s_study = sorted(
        frame.groupby("study")["spoken"]
        .apply(lambda s: s.notna().any())
        .loc[lambda s: ~s]
        .index
    )
    never_s_form = sorted(
        frame.groupby("survey_vocab_max")["spoken"]
        .apply(lambda s: s.notna().any())
        .loc[lambda s: ~s]
        .index
    )
    print(f"spoken missing, STRUCTURAL (study never records it): "
          f"{int((~s_obs & ~study_measures_s).sum())}")
    print(f"    studies that never record a spoken count: {never_s_study}")
    print(f"    forms that never record a spoken count:   {never_s_form}")
    print(f"spoken missing, OTHER: {int((~s_obs & study_measures_s).sum())}")
    print(
        frame.loc[~s_obs & study_measures_s]
        .groupby(["study", "survey_vocab_max"])
        .size()
        .rename("rows")
        .to_frame()
        .to_string()
    )
    print()


def account_for_the_support(frame: pd.DataFrame, definition) -> None:
    print("== 2. What carries the lag, and why the rest does not ==\n")
    prev_idx, lagged, in_spoken, conditional = _lag(frame, definition)
    supporting = lagged & in_spoken
    subject = frame["subject_code"].to_numpy()
    age = frame["age"].to_numpy()

    print(f"rows with a prior-wave understood source: {int(lagged.sum())}")
    print(f"... entering the spoken likelihood:       {int(supporting.sum())}"
          f"  ({frame.loc[supporting, 'subject_code'].nunique()} children, "
          f"{frame.loc[supporting, 'study'].nunique()} studies)")
    print(f"    on the conditional S|U branch: {int((supporting & conditional).sum())}")
    print(f"    on the marginal fallback:      "
          f"{int((supporting & ~conditional).sum())}")
    form_measures_u = _measures(frame, "understood", "survey_vocab_max").to_numpy()
    fallback = supporting & ~conditional
    print(f"        of those, on a form that never measures comprehension: "
          f"{int((fallback & ~form_measures_u).sum())}")

    first_age = pd.Series(age).groupby(subject).transform("min").to_numpy()
    at_first_wave = age == first_age
    print(f"\nrows with no lag: {int((~lagged).sum())}")
    print(f"... at the child's first wave (no earlier wave exists): "
          f"{int((~lagged & at_first_wave).sum())}")
    later = ~lagged & ~at_first_wave
    print(f"... later, but every earlier wave lacked comprehension: {int(later.sum())}")
    print(frame.loc[later].groupby("study").size().rename("rows").to_frame().to_string())

    waves = frame.groupby("subject_code")["age"].nunique()
    per_child = (
        frame.assign(supports=supporting)
        .groupby("subject_code")
        .agg(
            u_any=("understood", lambda s: s.notna().any()),
            s_any=("spoken", lambda s: s.notna().any()),
            supports=("supports", "any"),
        )
        .join(waves.rename("waves"))
    )
    multi = per_child["waves"] > 1
    print(f"\nchildren: {len(per_child)}")
    print(f"... single-wave, so they can never carry a lag: {int((~multi).sum())}")
    print(f"... multi-wave and supporting:                  "
          f"{int((multi & per_child['supports']).sum())}")
    never = multi & ~per_child["supports"]
    print(f"... multi-wave and never supporting:            {int(never.sum())}")
    print(f"        never any understood: {int((never & ~per_child['u_any']).sum())}")
    print(f"        never any spoken:     {int((never & ~per_child['s_any']).sum())}")
    print(f"        both, never in order: "
          f"{int((never & per_child['u_any'] & per_child['s_any']).sum())}")

    gap = (age - age[prev_idx])[supporting]
    target = age[supporting]
    print(f"\ntarget ages: median {np.median(target):.0f}, "
          f"IQR {np.percentile(target, 25):.0f}-{np.percentile(target, 75):.0f}, "
          f"range {target.min():.0f}-{target.max():.0f}")
    print(f"gaps: median {np.median(gap):.0f}, "
          f"IQR {np.percentile(gap, 25):.0f}-{np.percentile(gap, 75):.0f}, "
          f"range {gap.min():.0f}-{gap.max():.0f}")
    for threshold in (12, 18, 24):
        print(f"    gap > {threshold:>2} months: {int((gap > threshold).sum())} rows "
              f"({100 * (gap > threshold).mean():.1f}%)")

    print("\nsupporting rows and children by study:")
    contribution = (
        frame.assign(supporting=supporting)
        .groupby("study")
        .agg(rows=("age", "size"), supporting=("supporting", "sum"))
    )
    contribution["children"] = (
        frame.loc[supporting].groupby("study")["subject_code"].nunique()
    )
    contribution["pct_of_support"] = (
        100 * contribution["supporting"] / contribution["supporting"].sum()
    ).round(1)
    print(contribution.fillna(0).to_string())
    print()


def probe_ignorability(frame: pd.DataFrame) -> None:
    print("== 3. Is the missingness ignorable given study, age and the child? ==\n")
    measures_u = _measures(frame, "understood", "survey_vocab_max")
    frame = frame.assign(measures_u=measures_u)
    switchers = sorted(
        frame.groupby("study")["measures_u"].nunique().loc[lambda s: s > 1].index
    )
    print(f"studies using both a comprehension-measuring and a comprehension-less "
          f"form: {switchers}\n")

    # (a) The parallel-form artefact, measured on the waves where it lives.
    mixed = (
        frame.groupby(["subject_code", "age"])["measures_u"]
        .nunique()
        .loc[lambda s: s > 1]
        .index
    )
    pairs = []
    for code, age in mixed:
        wave = frame[(frame["subject_code"] == code) & (frame["age"] == age)]
        short = wave[wave["measures_u"]]["spoken"]
        long = wave[~wave["measures_u"]]["spoken"]
        if short.notna().any() and long.notna().any():
            pairs.append(long.max() - short.max())
    pairs = pd.Series(pairs, dtype=float)
    print(f"(a) waves carrying both form classes: {len(mixed)}; with a spoken count "
          f"on each: {len(pairs)}")
    if len(pairs):
        print(f"    the longer form records a median of {pairs.median():+.0f} words "
              f"more; it is higher on {int((pairs > 0).sum())} of {len(pairs)} pairs")
    print("    -> an item-pool difference of this size cannot produce the "
          "between-child gaps below.\n")

    # (b) Between children, with those waves removed so (a) cannot contribute.
    index = pd.MultiIndex.from_arrays([frame["subject_code"], frame["age"]])
    solo = frame[~index.isin(mixed) & frame["spoken"].notna()].copy()
    solo["band"] = pd.cut(solo["age"], bins=AGE_BANDS, right=False)
    print("(b) within a study and an age band, rank correlation between being on "
          "the\n    comprehension-less form and the spoken count:")
    for study in switchers:
        rows = solo[solo["study"] == study]
        if rows["measures_u"].nunique() < 2:
            continue
        print(f"    {study}:")
        for band, group in rows.groupby("band", observed=True):
            if group["measures_u"].nunique() < 2 or len(group) < 15:
                continue
            rho = stats.spearmanr(
                (~group["measures_u"]).to_numpy(float), group["spoken"]
            ).statistic
            median = group.groupby("measures_u")["spoken"].median()
            print(f"        {str(band):<12} rho = {rho:+.2f}  (n = {len(group):>3}) "
                  f" median spoken {median.get(False, np.nan):>5.0f} on the "
                  f"comprehension-less form vs {median.get(True, np.nan):>5.0f}")
    print()

    # (c) The same comparison with age matched exactly, which is the version
    #     with no residual gradient inside a band -- and far too thin to settle.
    strata = []
    for (study, age), group in solo.groupby(["study", "age"]):
        if group["measures_u"].nunique() < 2:
            continue
        with_u = group.loc[group["measures_u"], "spoken"]
        without = group.loc[~group["measures_u"], "spoken"]
        if len(with_u) >= 2 and len(without) >= 2:
            strata.append(
                {
                    "study": study,
                    "age": age,
                    "n_measures_u": len(with_u),
                    "n_no_u": len(without),
                    "median_measures_u": with_u.median(),
                    "median_no_u": without.median(),
                    "difference": without.median() - with_u.median(),
                }
            )
    strata = pd.DataFrame(strata)
    print("(c) exactly matched study and recorded age, both classes with >= 2 rows:")
    if strata.empty:
        print("    no stratum qualifies")
        return
    print(strata.to_string(index=False))
    higher = int((strata["difference"] > 0).sum())
    print(f"\n    {len(strata)} strata; the comprehension-less form is higher in "
          f"{higher} and lower in {int((strata['difference'] < 0).sum())}")
    print(f"    median of the within-stratum differences: "
          f"{strata['difference'].median():+.0f} words")
    if len(strata) >= 6:
        test = stats.wilcoxon(strata["difference"])
        print(f"    Wilcoxon signed-rank: statistic {test.statistic:.1f}, "
              f"p = {test.pvalue:.3f}")
    print("    -> directional, not decisive: these strata hold a handful of rows "
          "each.\n")


def support_under_each_restriction() -> None:
    print("== 4. What each registered restriction leaves of the support ==\n")
    rows = {}
    for name in ["baseline", *sorted(variants_for("vg16"))]:
        if name == "baseline":
            definition = MODEL_REGISTRY["vg16"]
        else:
            (definition,) = build_variant("vg16", name)
        frame = _frame(definition)
        _, lagged, in_spoken, conditional = _lag(frame, definition)
        supporting = lagged & in_spoken
        rows[name] = {
            "frame_rows": len(frame),
            "frame_studies": frame["study"].nunique(),
            "supporting": int(supporting.sum()),
            "children": int(frame.loc[supporting, "subject_code"].nunique()),
            "studies": int(frame.loc[supporting, "study"].nunique()),
        }
    print(pd.DataFrame(rows).T.to_string())
    print()


def main() -> None:
    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", 300)
    definition = MODEL_REGISTRY["vg16"]
    frame = _frame(definition)
    assert len(frame) == EXPECTED_ROWS, (
        f"VG16's frame has {len(frame)} rows, not the {EXPECTED_ROWS} the fit of "
        "record was made on: the loader rules have moved and these counts are "
        "about a different dataset."
    )
    assert frame["subject_code"].nunique() == EXPECTED_CHILDREN

    classify_missingness(frame)
    account_for_the_support(frame, definition)
    probe_ignorability(frame)
    support_under_each_restriction()


if __name__ == "__main__":
    main()
