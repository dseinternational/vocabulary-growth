# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import os

import duckdb
import pandas as pd

import vocab_growth.environment as local_env

# The widened language scopes are defined with the model definitions (data_utils
# imports from definitions, not the other way round), but they belong to the loader's
# vocabulary as far as callers are concerned. Re-exported explicitly: the ``X as X``
# form marks a deliberate re-export, so it is not pruned as an unused import.
from vocab_growth.models.definitions import (
    ENGLISH_AND_ROMANCE_LANGUAGES as ENGLISH_AND_ROMANCE_LANGUAGES,
)
from vocab_growth.models.definitions import (
    ENGLISH_LANGUAGES,
    Population,
)
from vocab_growth.models.definitions import (
    ROMANCE_LANGUAGES as ROMANCE_LANGUAGES,
)

VOCABULARY_DATA_PATH = os.path.join(local_env.DATA_DIR, "vocabulary.duckdb")

WORDBANK_BIVARIATE_FORMS = ("Oxford CDI", "WG")
"""Wordbank forms whose comprehension is an independent measurement.

On the other English forms (WS, WSShort, TEDS Twos/Threes) the
``comprehension`` column is a production proxy (``comprehension ==
production`` by data convention), so only these forms may contribute
``understood`` observations. This is a property of the instrument, not the
population: the guard applies both to the TD loader (:func:`load_data`) and
to the us_01/Edgin DS block of the ``vocab_combined`` view
(:func:`vocab_combined_view_sql`). See
``notes/202605151630-vg06-ws-comprehension-issue.md`` (TD) and
``notes/202607061200-us01-edgin-ws-comprehension-issue.md`` (DS).
"""

WORDBANK_SPOKEN_ONLY_FORMS = ("WS",)
"""Wordbank forms that contribute production observations only."""

WORDBANK_FORM_ITEMS: dict[tuple[str, str], int] = {
    ("English (American)", "WG"): 396,
    ("English (British)", "Oxford CDI"): 416,
    ("Italian", "WG"): 408,
    ("Spanish (European)", "WG"): 309,
    ("Catalan", "WG"): 423,
    ("Portuguese (European)", "WG"): 317,
}
"""Word-item counts of the Wordbank comprehension forms, keyed by ``(language, form)``.

The typically-developing loader carries no ``survey_vocab_max`` -- Wordbank's
by-child export does not record one -- so this is the ceiling a count on each of
these forms can be expressed against. The counts come from each instrument's
definition file in ``langcog/wordbank`` (``type == "word"`` rows; the method
reproduces English (American) 396 exactly), as tabulated in
``notes/202608031500-td-romance-extension.md``, with one deliberate exception: the
Oxford CDI is entered at **416**, the ceiling every Oxford-form source in
``vocab_combined`` carries (see the native-ceiling comment beside the view SQL),
where the definition file counts 418 word rows. The two unadmitted Romance forms
(Catalan, Portuguese) are included so the cross-language nesting check can be rerun
on the candidate set that note reported. Consumer:
:func:`vocab_growth.descriptive.td_form_alignment_table`.
"""

US01_WS_VOCAB_MAX = 680
"""Native vocabulary ceiling of the us_01 Words & Sentences form."""

TD_POOL_EXCLUDED_DATASETS = ("Edgin",)
"""Wordbank datasets excluded from the typically developing reference pool.

``Edgin`` supplies the ``us_01`` Down syndrome source. The July export audit
found two rows meeting the TD filter despite cohort and preparation concerns,
including a Words & Sentences ceiling record in a suspect batch. Excluding the
dataset keeps these records out of the reference used to assess DS exclusions.
The small row count alone does not establish that estimates are unchanged.

``us_01`` now uses item-level contributor files rather than this export. See
``data/vocab_data_us_01.md`` and
``notes/202607261245-edgin-duplicated-outcome-records.md`` section 13.
"""

TD_POOL_AGE_MONTHS = (8, 30)
"""Age window, in months, of the typically developing reference pool.

The window keeps admitted forms inside the TD models' shared 8-30-month GP
domain. Italian Words & Gestures starts at 7 months, but extending the shared
domain for its few younger records would change existing model definitions.

``max_age_months`` overrides the upper bound per model. The lower bound has no
per-model override. Outcome-specific form restrictions and reporting caps can
narrow this window further, particularly for comprehension.
"""


SIGNED_ONLY_STUDIES = ("uk_01",)
"""Studies whose ``signed`` field excludes words that are also spoken.

The signing models estimate total sign use, so these fields are not comparable
without item-level re-derivation.  Keep the source rows for understood/spoken
outcomes while masking only their ``signed`` value by default.
"""

UNCERTAIN_SIGN_STUDIES: tuple[str, ...] = ()
"""Studies whose signing-field construct has not yet been source-verified.

Empty since 2026-08-12. It held ``uk_06`` from 16 July, when its inclusion was
reversed pending confirmation that its signing field measured total sign use rather
than uk_01's sign-only construct (issue #211).

The source has now confirmed uk_06 used the **standard DSE checklists, as in
ie_01/ie_02**, whose completion instructions make each of columns 2-5 conditional on
comprehension -- column 2 is "understands and signs (tick for imitated signs as well
as for spontaneous signs)". That is a *total* sign count, so uk_06 is comparable with
uk_02, nz_01, es_01 and uk_07 and needs no mask. The committed data agrees on every
row: ``signed`` and ``spoken`` are both nested within ``understood`` 11 times out of
11, and ``signed + spoken`` exceeds ``understood`` on 7 of 11 -- impossible under a
mutually exclusive reading, exactly as overlapping per-word ticks predict.

The constant is kept rather than deleted: it is the mechanism for the next source
whose construct is unverified, and emptying it records that this one was resolved by
evidence rather than quietly dropped. See data/vocab_data_uk_06.md."""

INCOMPLETE_ADMINISTRATION_CEILINGS: dict[str, tuple[int, ...]] = {"ie_01": (460,)}
"""Per-study ``survey_vocab_max`` values marking a partial administration.

An administration that omitted part of the reference inventory does not produce a
count on the 810-item scale the model likelihoods score against, so its counts are
masked by default.

This is a different situation from the shorter MacArthur-derived forms (Oxford CDI
416, MB-CDI WG 396, NZCDI 675). Those are *nested* instruments whose absent items
are the rarer, later-acquired words an ability-matched child mostly does not know,
and a dual-form crosswalk fitted to the uk_02 children who took both the DSE and
Oxford forms put the fixed-810 count ratio near 1 across the range where the short
forms are administered (see ``notes/202607121200-statistical-model-review.md``
§3A). Here, by contrast, a whole 350-item subscale of the *same* instrument was
not administered:

- ``ie_01`` baseline wave (ceiling 460 = DSE Checklists 1 + 2). Checklist 3 is
  recorded as exactly zero for all 59 children on all three response types
  (understood, imitates, says); no baseline total exceeds 460; and 33 of 46
  follow-up records carry non-zero Checklist 3 counts up to 328, including
  children whose baseline total already exceeded 390. At matched vocabulary the
  follow-up wave puts about 9.5% of Checklist 3 known, against 0% at baseline —
  so the zeros are an un-administered subscale, not ability.
"""

DSE_SHORT_FORM_CEILINGS: dict[str, tuple[int, ...]] = {"ie_02": (476,)}
"""Per-study ``survey_vocab_max`` values marking a DSE short form kept on the 810 scale.

``ie_02`` administered DSE Checklists 1 and 2 only (127 + 349 = 476 achievable
words); Checklist 3 was not given. That is the same instrument subset as
``ie_01``'s baseline wave, which :data:`INCOMPLETE_ADMINISTRATION_CEILINGS` masks.
The study owner decided on 2026-09-15 to treat ``ie_02`` differently: its counts
stay in the pool on the 810-item reference scale, as the nested Oxford CDI and
MB-CDI forms do, and its ceiling is recorded as 476 so the rules that read
``survey_vocab_max`` see the form it actually was. The form-ceiling guard therefore
drops the administrations whose ``understood`` of 477 exceeds it, and
:func:`restrict_to_dse_native_administrations` no longer counts ``ie_02`` as native.

The evidence the decision rested on, measured on ``ie_01``'s follow-up wave, the
one wave in the pool with all three checklists recorded (44 comprehension
records): Checklist 3 adds little below about 300 words on Checklists 1 + 2
(medians of 0 to 24 words per band), but a median of 100 at 300-400 and 233 at
400-476, where it is 22-35% of the child's full count. The bands hold 5 to 11
records each. In ``ie_02``, 17.5% of comprehension counts are at or above 300 and
the spoken counts are almost all small (90th percentile 42), so masking the study
would have discarded mostly unaffected rows -- 65 children, and 111 of the joint
models' 251 signed observations -- to remove an understatement concentrated in a
fifth of one outcome. (Those figures are the pool at the decision; the guard's
one dropped administration makes them 110 of 250 signed observations after it.)

The understatement is real where it occurs, so it has a registered check rather
than none: :func:`mask_short_form_comprehension`, reached through the
``mask_dse_short_form_comprehension`` definition field, masks these studies'
comprehension counts for a sensitivity arm. The asymmetry with ``ie_01``'s baseline
is deliberate but not argued from evidence that separates the two, and is recorded
as such.
"""

DUPLICATED_OUTCOME_MAX_AGE_MONTHS = 18
DUPLICATED_OUTCOME_MIN_UNDERSTOOD = 100
DUPLICATED_OUTCOME_RATIO = 0.75
"""Screening rule for infant administrations with unusually similar outcomes.

Mask when ``spoken >= DUPLICATED_OUTCOME_RATIO * understood``,
``understood >= DUPLICATED_OUTCOME_MIN_UNDERSTOOD`` and
``age <= DUPLICATED_OUTCOME_MAX_AGE_MONTHS``. All conditions are required;
similar outcome counts at older ages are not flagged by this rule.

The July ``us_01`` audit found a separate high-ratio cluster. The 0.75 threshold
falls in the observed gap between ratios of about 0.55 and 0.86; it is a
sample-specific screening choice, not a developmental limit. External DS counts
and the same children's other administrations added concern, but cannot prove
which response column is wrong or that every flagged record is defective.

The rule masks understood, spoken and produced counts because aggregate totals
cannot identify a reliable column. High-comprehension records with an ordinary
production gap are retained. ``include_duplicated_outcomes=True`` reinstates
flagged values for sensitivity analysis. See
``notes/202607261245-edgin-duplicated-outcome-records.md`` and the current
``data/vocab_data_us_01.md`` source record.
"""


UK07_WITHHELD_ADMINISTRATIONS: tuple[tuple[str, str], ...] = (
    ("ID_44BA6806E829CE6B", "t3"),
)
"""uk_07 administrations withheld pending clarification with the source team.

Keyed by ``(subject_id, timepoint)``. One administration is listed: at 58 months
this child records 191 words understood against 489 produced. It is the only row
in the source where production exceeds comprehension, and it sits at the end of a
reported comprehension decline (349 → 291 → 191 across the three visits) while
production rises (185 → 263 → 489). The likeliest reading is a parent-report
artefact — only the expressive columns completed at the later visit — but that is
a hypothesis about how the form was filled in, not something the aggregate counts
can settle.

Withheld here rather than left to the general rule. Since 2026-08-25 the
comparable records in ``ie_01`` and ``it_01`` are masked by
:func:`mask_comprehension_below_production` -- previously they were retained and
flagged, and this docstring drew the contrast against that -- and since
2026-09-13 one in ``uk_02`` as well, nine in all. (Two ``uk_01`` records were
masked too until 2026-09-14, when the source's comprehension count was found to
exclude words the child also says and was corrected upstream.) The reason for
keeping a separate mechanism is unchanged: those are a known, stable property
of closed sources, whereas this one is an open question with a reachable source
team, so the row is held out of the prepared data entirely until the study owner
has an explanation, rather than reaching ``vocab_combined`` and being masked with
a reinstatement flag. Removing the
entry from this tuple and re-running ``scripts/prepare_data.py`` reinstates it.
This is the same treatment as the ie_02 subject excluded in ``prepare_data.py``.

Applied at CSV load, so the row is absent from the ``vocab_uk_07`` table, the
``vocab_combined`` view, ``vocab_data_merged.csv`` and VG15's cross-tab path
alike — there is no route by which a model can see it.
"""


def drop_uk07_withheld_administrations(
    raw: pd.DataFrame,
    *,
    subject_col: str = "subject_id",
) -> tuple[pd.DataFrame, int]:
    """Drop the uk_07 rows listed in :data:`UK07_WITHHELD_ADMINISTRATIONS`.

    Takes the raw uk_07 CSV frame (``subject_id`` and ``timepoint`` columns) and
    returns it without the withheld administrations, plus the number removed.
    Both readers of that CSV — ``scripts/prepare_data.py`` and the VG15
    cross-tab loader — call this, so neither can drift from the other.
    """
    required = {subject_col, "timepoint"}
    missing = required - set(raw.columns)
    if missing:
        raise KeyError(
            "uk_07 withholding requires columns: " + ", ".join(sorted(missing))
        )

    keys = pd.MultiIndex.from_arrays([raw[subject_col], raw["timepoint"]])
    drop = keys.isin(UK07_WITHHELD_ADMINISTRATIONS)
    return raw.loc[~drop].reset_index(drop=True), int(drop.sum())


UK01_WITHHELD_SUBJECTS: tuple[str, ...] = ("ID_E33ADE657109EBB8",)
"""uk_01 subjects withheld as probable homonym fusions of two children.

uk_01 is the one source whose subject identifier is derived from the child's
*name* alone (see ``prepare/uk_01_edg.py`` in the ``research-data-analysis``
repository): the raw SPSS file carries no per-child identifier, so the name is
the longitudinal linker, and two different children who share a name are
silently fused into one ``subject_id``. That risk is documented at source as
the homonym caveat; this constant records the one id where the fused pattern
is actually observed.

``ID_E33ADE657109EBB8`` (F) carries four administrations that split perfectly
into two interleaved, internally consistent modality profiles — a signer who
barely speaks and a speaker who never signs::

    age 66 (WG):  spoken   8, signed 225
    age 76 (WS):  spoken 451, signed   0
    age 78 (WS):  spoken  27, signed 126
    age 88 (WS):  spoken 483, signed   0

Read as one child this is a 424-word production collapse in two months followed
by a 456-word surge — the "uk_01 record at 76 months" the longitudinal-collapse
rule (:data:`COLLAPSE_FACTOR`) deliberately left for separate investigation.
Read as two children ({66, 78} and {76, 88}) both trajectories are ordinary.
The four rows sit under one exact canonicalised name in the raw source
(verified 2026-08-31), which carries no date of birth, record number or any
other disambiguator, so the split cannot be made mechanically — and assigning
rows to children by their outcome profile would be selection on the outcome.
The whole id is therefore withheld, all four rows, pending adjudication
against the original study records.

Deliberately *not* withheld: ``ID_CEBD1F6C4348C78C`` (M, eight rows, 35–80
months), the only other uk_01 id pairing a substantial-signer row with a
non-signing-speaker row. Its profile — a heavy signer at 35 months becoming a
240-word non-signing speaker by 44 — is also consistent with a genuine
sign-to-speech transition, which is what the signing models estimate, so it
stays in as a sensitivity target rather than an exclusion.

Applied at CSV load in ``scripts/prepare_data.py``, so the rows are absent
from the ``vocab_uk_01`` table, the ``vocab_combined`` view and
``vocab_data_merged.csv`` alike. In the default pool the cost is four spoken
observations: uk_01's ``signed`` is already masked by default
(:data:`SIGNED_ONLY_STUDIES`) and ``understood`` is missing on all four rows.
Removing the id from this tuple and re-running ``scripts/prepare_data.py``
reinstates it. See ``notes/202608311600-uk01-homonym-fusion.md``.
"""


def drop_uk01_withheld_subjects(
    raw: pd.DataFrame,
    *,
    subject_col: str = "subject_id",
) -> tuple[pd.DataFrame, int]:
    """Drop every row of the uk_01 subjects listed in :data:`UK01_WITHHELD_SUBJECTS`.

    Takes the raw uk_01 CSV frame and returns it without the withheld subjects'
    administrations, plus the number of rows removed. Withholding is by whole
    subject rather than by administration because the defect is the identifier
    itself: which rows belong to which child is exactly what cannot be
    recovered from the aggregate data.
    """
    if subject_col not in raw.columns:
        raise KeyError(f"uk_01 withholding requires column: {subject_col}")
    drop = raw[subject_col].isin(UK01_WITHHELD_SUBJECTS)
    return raw.loc[~drop].reset_index(drop=True), int(drop.sum())


IE02_WITHHELD_ADMINISTRATIONS: tuple[tuple[str, str], ...] = (
    ("ID_62C63BE2B3B627E6", "t2"),
)
"""ie_02 administrations withheld pending source clarification.

Keyed by ``(subject_id, timepoint)``. The listed t2 record reports 442 words
understood, 3 spoken and 301 signed at 48 months, against 111 understood,
72 spoken and 64 signed three months earlier. The combined changes raised
concern about differing checklist completion. Aggregate counts cannot establish
whether the changes are real or which column is reliable, so the whole t2
administration is withheld; t1 remains.

``scripts/prepare_data.py`` drops the row at CSV load, before the database,
merged CSV and joint-model paths. Remove the tuple entry and rebuild prepared
data to reinstate it. See ``notes/202608311830-steep-within-child-gains.md``.
"""


def drop_ie02_withheld_administrations(
    raw: pd.DataFrame,
    *,
    subject_col: str = "subject_id",
) -> tuple[pd.DataFrame, int]:
    """Drop the ie_02 rows listed in :data:`IE02_WITHHELD_ADMINISTRATIONS`.

    Takes the raw ie_02 CSV frame (``subject_id`` and ``timepoint`` columns)
    and returns it without the withheld administrations, plus the number
    removed. The child's other timepoint is retained.
    """
    required = {subject_col, "timepoint"}
    missing = required - set(raw.columns)
    if missing:
        raise KeyError(
            "ie_02 withholding requires columns: " + ", ".join(sorted(missing))
        )

    keys = pd.MultiIndex.from_arrays([raw[subject_col], raw["timepoint"]])
    drop = keys.isin(IE02_WITHHELD_ADMINISTRATIONS)
    return raw.loc[~drop].reset_index(drop=True), int(drop.sum())


def mask_incomparable_signed_outcomes(
    df: pd.DataFrame,
    *,
    include_signed_only: bool = False,
    include_uncertain: bool = False,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Mask signing fields that do not identify comparable total sign use.

    The returned frame is a copy.  Understood and spoken observations from the
    affected studies are retained; only ``signed`` is set missing.  The counts
    report how many observed signing values were removed from each study so fit
    logs and provenance make the source restriction explicit.
    """
    required = {"study", "signed"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(
            "Signing-source harmonisation requires columns: "
            + ", ".join(sorted(missing))
        )

    out = df.copy()
    excluded: list[str] = []
    if not include_signed_only:
        excluded.extend(SIGNED_ONLY_STUDIES)
    if not include_uncertain:
        excluded.extend(UNCERTAIN_SIGN_STUDIES)

    dropped: dict[str, int] = {}
    for study in excluded:
        mask = (out["study"] == study) & out["signed"].notna()
        dropped[study] = int(mask.sum())
        out.loc[mask, "signed"] = float("nan")
    return out, dropped


def mask_incomplete_administrations(
    df: pd.DataFrame,
    *,
    include_incomplete: bool = False,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Mask counts from administrations that omitted part of the reference inventory.

    Every model likelihood scores counts against the common 810-item inventory, so
    a count from a partial administration understates the child's reference-scale
    vocabulary by however much of the inventory went unasked. Rescaling is not
    available either: the omitted DSE Checklist 3 items are markedly harder than
    the administered ones, so the proportion known on Checklists 1-2 is not the
    proportion known on all three.

    The affected rows are identified by their recorded ``survey_vocab_max`` (see
    :data:`INCOMPLETE_ADMINISTRATION_CEILINGS`) and their outcome columns are set
    missing; the rows are retained so age coverage and provenance stay auditable.
    The returned counts report how many observed values were masked per study, for
    the fit log. Pass ``include_incomplete=True`` to reintroduce them as a
    sensitivity.
    """
    required = {"study", "survey_vocab_max"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(
            "Incomplete-administration masking requires columns: "
            + ", ".join(sorted(missing))
        )

    out = df.copy()
    dropped: dict[str, int] = {}
    if include_incomplete:
        return out, dropped

    outcome_columns = [
        column
        for column in ("understood", "spoken", "signed", "produced")
        if column in out.columns
    ]
    for study, ceilings in INCOMPLETE_ADMINISTRATION_CEILINGS.items():
        mask = out["study"].eq(study) & out["survey_vocab_max"].isin(ceilings)
        if not mask.any():
            continue
        dropped[study] = int(out.loc[mask, outcome_columns].notna().to_numpy().sum())
        out.loc[mask, outcome_columns] = float("nan")
    return out, dropped


def mask_short_form_comprehension(
    df: pd.DataFrame,
    *,
    mask_short_form: bool = False,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Mask comprehension counts from DSE short-form administrations, as a sensitivity.

    The inverse of the other rules in this module: by default nothing is masked,
    because :data:`DSE_SHORT_FORM_CEILINGS` keeps those administrations on the
    810-item scale. With ``mask_short_form=True`` their ``understood`` counts are
    set missing -- the outcome where the omitted checklist's harder words matter --
    and every other count and the rows themselves are kept. The returned counts
    report how many comprehension values were masked per study, for the fit log.
    """
    required = {"study", "survey_vocab_max"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(
            "Short-form comprehension masking requires columns: "
            + ", ".join(sorted(missing))
        )

    out = df.copy()
    masked: dict[str, int] = {}
    if not mask_short_form or "understood" not in out.columns:
        return out, masked

    for study, ceilings in DSE_SHORT_FORM_CEILINGS.items():
        mask = out["study"].eq(study) & out["survey_vocab_max"].isin(ceilings)
        count = int(out.loc[mask, "understood"].notna().sum())
        if count == 0:
            continue
        masked[study] = count
        out.loc[mask, "understood"] = float("nan")
    return out, masked


def mask_duplicated_outcome_administrations(
    df: pd.DataFrame,
    *,
    include_duplicated: bool = False,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Mask infant administrations matching the similar-outcome screening rule.

    The rule is documented on ``DUPLICATED_OUTCOME_RATIO``. Understood, spoken
    and produced values are masked because aggregate counts cannot identify a
    reliable column. The rule does not prove that columns were duplicated.
    Rows remain for provenance, and returned counts record masked values per
    study. ``include_duplicated=True`` reinstates values for sensitivity analysis.
    """
    required = {"age", "understood", "spoken"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(
            "Duplicated-outcome masking requires columns: "
            + ", ".join(sorted(missing))
        )

    out = df.copy()
    dropped: dict[str, int] = {}
    if include_duplicated:
        return out, dropped

    understood = pd.to_numeric(out["understood"], errors="coerce")
    spoken = pd.to_numeric(out["spoken"], errors="coerce")
    age = pd.to_numeric(out["age"], errors="coerce")
    suspect = (
        understood.notna()
        & spoken.notna()
        & age.notna()
        & (age <= DUPLICATED_OUTCOME_MAX_AGE_MONTHS)
        & (understood >= DUPLICATED_OUTCOME_MIN_UNDERSTOOD)
        & (spoken >= DUPLICATED_OUTCOME_RATIO * understood)
    )
    if not suspect.any():
        return out, dropped

    outcome_columns = [
        column
        for column in ("understood", "spoken", "produced")
        if column in out.columns
    ]
    study_labels = out["study"] if "study" in out.columns else pd.Series("", index=out.index)
    for study, count in (
        out.loc[suspect, outcome_columns].notna().sum(axis=1).groupby(study_labels[suspect]).sum().items()
    ):
        dropped[str(study)] = int(count)
    out.loc[suspect, outcome_columns] = float("nan")
    return out, dropped


COMPREHENSION_BELOW_PRODUCTION_STUDIES: tuple[str, ...] = ("ie_01", "it_01", "uk_02")
"""Studies with documented comprehension-below-production records.

Inclusive comprehension should contain the words produced. The masking function
checks every row against ``max(produced, spoken)``, with missing values skipped;
this tuple documents affected studies rather than restricting the function.
Only comprehension is masked, and equality is retained.

Use the maximum, never ``spoken + signed``: speech and signing can overlap.
``produced`` is a union in some sources and speech alone in others. ``signed``
is omitted from this bound because its relation to ``produced`` varies by source.

The ie_01 comprehension field has documented source concerns. A uk_01 coding
error once caused this signature but was corrected upstream on 2026-09-14,
so uk_01 is no longer listed. The uk_02 record lacks a produced union, but its
spoken count alone exceeds comprehension. Retaining usable production while
masking comprehension is the study's handling decision, not proof that every
retained production count is correct.

``UK07_WITHHELD_ADMINISTRATIONS`` separately removes an open source-query record
before this rule sees it. ``include_comprehension_below_production=True``
reinstates values masked here for sensitivity analysis.
"""


#: Studies whose older administrations are a structurally distinct sub-sample
#: rather than legitimately older children.
#:
#: The Down syndrome pool deliberately **admits** administrations above a form's
#: registered age window -- for this population an early-vocabulary form given to
#: an older child is developmentally appropriate, and those rows are us_01's only
#: comprehension observations between 19 and 27 months. So age alone must never
#: be the criterion, and this rule is stated on *provenance* instead, exactly as
#: :data:`CEILING_ONLY_CHILD_STUDIES` is.
#:
#: ``us_03``: four children (workbook ``id`` 1-5, one of which carries no CDI
#: data) sit at 61-80 months against 17-35 for all 286 other administrations --
#: a 26-month gap with nothing in it. (62-80 and 27 before 2026-09-15, when ages
#: were rounded rather than floored to complete months; the bound below separates
#: both.) They have no second visit, sit near the
#: form's ceiling at 286-376 produced of 396, and their ages are the only ones in
#: the file not recorded as a whole hundredth of a year. The source's own
#: documentation concludes they came from a different file, plausibly the second
#: of the two projects its citation names, and records that the workbook carries
#: no project identifier to confirm it with (``data/vocab_data_us_03.md``).
#:
#: What makes this an exclusion rather than a caveat is the measurement, not the
#: age: whether those four were given the same 396-word Words and Gestures form
#: is unknown, and ``survey_vocab_max`` -- the denominator every likelihood in
#: this project divides by -- is the least certain value in the source for
#: exactly those rows. One of them records 432 understood against a 396-item
#: ceiling, which is in range on a larger form and impossible on this one.
#: Admitting them means asserting a denominator the source will not support.
#:
#: The age bound is the empirical gap, not a developmental claim: it separates
#: the sub-sample and nothing else. Reinstate with
#: ``include_structurally_distinct_subsamples=True``; #289 task 0.3 records the
#: question to the data providers that would settle it properly.
STRUCTURALLY_DISTINCT_SUBSAMPLES: dict[str, float] = {"us_03": 35.0}


def drop_structurally_distinct_subsamples(
    df: pd.DataFrame,
    *,
    include_structurally_distinct_subsamples: bool = False,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Drop administrations from a documented structurally distinct sub-sample.

    Applies the rule documented on
    :data:`STRUCTURALLY_DISTINCT_SUBSAMPLES`. The whole administration goes,
    unlike the comprehension rules, because what is in doubt is the form -- and
    therefore ``survey_vocab_max`` -- rather than any one recorded count.
    """
    out = df.copy()
    dropped: dict[str, int] = {}
    if include_structurally_distinct_subsamples or "study" not in out.columns:
        return out, dropped

    age = pd.to_numeric(out.get("age"), errors="coerce")
    suspect = pd.Series(False, index=out.index)
    for study, above in STRUCTURALLY_DISTINCT_SUBSAMPLES.items():
        hit = (out["study"] == study) & age.notna() & (age > above)
        if hit.any():
            dropped[str(study)] = int(hit.sum())
        suspect |= hit
    if not suspect.any():
        return out, dropped
    return out.loc[~suspect].reset_index(drop=True), dropped


#: Sources whose ``produced`` union contains a non-vocal modality the source does
#: not separately record.
#:
#: VG18's outcome is the produced union and its covariate is sign group, so
#: ``signed`` is a component of its own outcome -- a tautology its module
#: docstring already states for the *union studies*, recommending either a
#: restricted study set or VG17's ``spoken`` outcome instead.
#:
#: ``us_03`` is worse than those, and differently. The union studies at least
#: record ``signed``, so their rows are grouped correctly and the confound is
#: visible in the contrast. ``us_03``'s expressive cell is "understands and says
#: **or signs**" (the study authors' wording; see ``data/vocab_data_us_03.md``)
#: with no separable sign component at all, so its rows are grouped ``unknown``
#: while their outcome silently contains the exposure. On the current frame that
#: is 284 rows into a reference group of 698 -- a 41% enlargement of the very
#: group the contrast is measured against, by rows whose outcome includes the
#: thing being contrasted.
#:
#: Applied to the produced outcome only. A source could record both a spoken
#: count and a union, and would then be perfectly usable for VG17's spoken
#: outcome; ``us_03`` contributes nothing there anyway, because it has no spoken
#: count to pass VG17's own filter.
PRODUCED_UNION_WITHOUT_SIGN_DETAIL: frozenset[str] = frozenset({"us_03"})


def drop_ungroupable_produced_unions(
    df: pd.DataFrame,
    *,
    include_ungroupable_produced_unions: bool = False,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Drop rows whose produced union hides the sign component being contrasted.

    Applies the rule documented on
    :data:`PRODUCED_UNION_WITHOUT_SIGN_DETAIL`. For a produced-outcome
    sign-group model only; the caller decides, because the same rows are
    unobjectionable for any model that does not condition on sign group.
    """
    out = df.copy()
    dropped: dict[str, int] = {}
    if include_ungroupable_produced_unions or "study" not in out.columns:
        return out, dropped
    hit = out["study"].isin(PRODUCED_UNION_WITHOUT_SIGN_DETAIL)
    if not hit.any():
        return out, dropped
    for study, count in hit[hit].groupby(out.loc[hit, "study"]).size().items():
        dropped[str(study)] = int(count)
    return out.loc[~hit].reset_index(drop=True), dropped


def mask_comprehension_below_production(
    df: pd.DataFrame,
    *,
    include_below_production: bool = False,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Mask comprehension counts that fall below the child's own production count.

    Applies the rule documented on
    :data:`COMPREHENSION_BELOW_PRODUCTION_STUDIES`. Only ``understood`` is
    masked; ``spoken``, ``signed`` and ``produced`` are left as recorded, and the
    row is retained so age coverage and provenance stay auditable.

    The comparison is against ``max(produced, spoken)``, skipping missing values,
    so a row whose ``produced`` was not recorded is still tested against its
    spoken count. ``signed`` is not a term; the constant's docstring says why.
    Requires a ``produced`` column all the same: the canonical loader always
    carries it, and a frame without it is a caller that has lost the union, not
    one this rule should quietly run on. Comparing against ``spoken + signed`` is
    wrong wherever the two modalities overlap, and is never substituted.

    The returned counts report how many comprehension values were masked per
    study, for the fit log. Pass ``include_below_production=True`` to reinstate
    them as a sensitivity.
    """
    required = {"understood", "produced"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(
            "Comprehension-below-production masking requires columns: "
            + ", ".join(sorted(missing))
        )

    out = df.copy()
    masked: dict[str, int] = {}
    if include_below_production:
        return out, masked

    understood = pd.to_numeric(out["understood"], errors="coerce")
    # The greatest recorded lower bound on words produced. `spoken` is contained
    # in production in every source, so it bounds the count from below even
    # where `produced` was not recorded. `signed` is deliberately not a term:
    # four sources record `produced` as the spoken count alone, so `signed`
    # exceeds it there and its relation to the column depends on the source.
    # A maximum skips missing values and cannot overstate production, where the
    # sum of the two modalities can and does.
    bound_columns = [column for column in ("produced", "spoken") if column in out.columns]
    production_bound = (
        out[bound_columns].apply(pd.to_numeric, errors="coerce").max(axis=1, skipna=True)
    )
    suspect = understood.notna() & production_bound.notna() & (understood < production_bound)
    if not suspect.any():
        return out, masked

    study_labels = (
        out["study"] if "study" in out.columns else pd.Series("", index=out.index)
    )
    for study, count in suspect[suspect].groupby(study_labels[suspect]).size().items():
        masked[str(study)] = int(count)
    out.loc[suspect, "understood"] = float("nan")
    return out, masked


IMPLAUSIBLE_PRODUCTION_CEILING_FRACTION = 0.9
IMPLAUSIBLE_PRODUCTION_MAX_AGE_MONTHS = 30
COLLAPSE_FACTOR = 5.0
COLLAPSE_MIN_VALUE = 50
"""Screening rules for near-ceiling counts and large recorded declines.

Both require the earlier or near-ceiling administration to have
``age <= IMPLAUSIBLE_PRODUCTION_MAX_AGE_MONTHS``. The age scope reflects the
source audit and external DS benchmark. A benchmark distribution does not
establish an impossible count or a universal developmental threshold.

Near-ceiling screening uses
``spoken >= IMPLAUSIBLE_PRODUCTION_CEILING_FRACTION * survey_vocab_max``.
The collapse rule flags a spoken count of at least ``COLLAPSE_MIN_VALUE`` when
an entry later in the child's age-sorted records is at most that count divided
by ``COLLAPSE_FACTOR``. The floor avoids flagging small absolute differences.
Recorded vocabulary can fall because of reporting, form changes or real loss;
the screening rule alone cannot identify the reason.

These rules mask spoken and produced, retaining comprehension. They follow
child-level ceiling and below-form-floor exclusions, and precede the separate
same-day-disagreement rule. Reinstating one rule's values may leave them masked
by another. ``count_reinstated_implausible_production`` measures the net change.

Older suspect profiles were investigated separately through
``UK01_WITHHELD_SUBJECTS`` and ``IE02_WITHHELD_ADMINISTRATIONS``. Read
``notes/202607261245-edgin-duplicated-outcome-records.md`` and the current
``data/vocab_data_us_01.md`` source record before changing scope or thresholds.
"""


def drop_duplicate_administrations(
    df: pd.DataFrame,
    *,
    subject_col: str = "subject_id",
) -> tuple[pd.DataFrame, int]:
    """Collapse rows that repeat the same measurement of the same child.

    ``us_01`` contains one administration recorded twice, identically (60 words
    understood and 1 spoken at 11 months). A repeated row double-weights that
    observation in every likelihood and, in the random-effect models, makes a
    single-visit child look like a repeated-measures one. Rows are matched on study,
    subject, age and every outcome present, so genuine repeat visits — which differ
    in age — are untouched.

    Returns the de-duplicated frame and the number of rows removed.
    """
    key = [column for column in ("study", subject_col, "age") if column in df.columns]
    if not key:
        raise KeyError("De-duplication requires at least one of: study, subject, age.")
    key += [
        column
        for column in ("understood", "spoken", "signed", "produced")
        if column in df.columns
    ]
    deduplicated = df.drop_duplicates(subset=key, keep="first")
    return deduplicated.reset_index(drop=True), len(df) - len(deduplicated)


def mask_implausible_production_administrations(
    df: pd.DataFrame,
    *,
    include_implausible: bool = False,
    subject_col: str = "subject_id",
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Mask production counts matching a near-ceiling or collapse signature.

    Applies both signatures documented on :data:`COLLAPSE_FACTOR`. Only the
    production-side outcomes are masked (``spoken`` and ``produced``); a paired
    ``understood`` value is left in place unless it too matches a signature,
    because on the Words & Sentences form ``understood`` is already absent by the
    production-proxy rule and on Words & Gestures the comprehension column is an
    independent measurement.

    Rows are retained so age coverage and provenance stay auditable. The returned
    counts report masked values per study. Pass ``include_implausible=True`` to
    reintroduce them as a sensitivity.
    """
    required = {"age", "spoken"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(
            "Implausible-production masking requires columns: "
            + ", ".join(sorted(missing))
        )

    out = df.copy()
    dropped: dict[str, int] = {}
    if include_implausible:
        return out, dropped

    age = pd.to_numeric(out["age"], errors="coerce")
    spoken = pd.to_numeric(out["spoken"], errors="coerce")
    in_window = age.notna() & (age <= IMPLAUSIBLE_PRODUCTION_MAX_AGE_MONTHS)

    suspect = pd.Series(False, index=out.index)
    if "survey_vocab_max" in out.columns:
        ceiling = pd.to_numeric(out["survey_vocab_max"], errors="coerce")
        suspect |= (
            in_window
            & spoken.notna()
            & ceiling.notna()
            & (spoken >= IMPLAUSIBLE_PRODUCTION_CEILING_FRACTION * ceiling)
        )

    if subject_col in out.columns:
        group_keys = out[subject_col].astype(str)
        if "study" in out.columns:
            group_keys = out["study"].astype(str) + "::" + group_keys
        for _, index in out.groupby(group_keys).groups.items():
            if len(index) < 2:
                continue
            ordered = index[age.loc[index].argsort()]
            values = spoken.loc[ordered]
            for position, row in enumerate(ordered):
                value = values.loc[row]
                if pd.isna(value) or value < COLLAPSE_MIN_VALUE or not in_window.loc[row]:
                    continue
                later = values.iloc[position + 1 :].dropna()
                if len(later) and later.min() * COLLAPSE_FACTOR <= value:
                    suspect.loc[row] = True

    if not suspect.any():
        return out, dropped

    outcome_columns = [
        column for column in ("spoken", "produced") if column in out.columns
    ]
    study_labels = (
        out["study"] if "study" in out.columns else pd.Series("", index=out.index)
    )
    counts = (
        out.loc[suspect, outcome_columns]
        .notna()
        .sum(axis=1)
        .groupby(study_labels[suspect])
        .sum()
    )
    dropped = {str(study): int(count) for study, count in counts.items()}
    out.loc[suspect, outcome_columns] = float("nan")
    return out, dropped


SAME_DAY_DISAGREEMENT_STUDIES: tuple[str, ...] = ("us_01",)
SAME_DAY_DISAGREEMENT_FACTOR = 5.0
SAME_DAY_DISAGREEMENT_MIN_VALUE = 100
"""Screening rule for large same-age production disagreements in us_01.

Within a ``(study, subject, age)`` group with at least two observed spoken
counts, mask a count if it is strictly above the group's minimum, at least
``SAME_DAY_DISAGREEMENT_MIN_VALUE`` and at least
``SAME_DAY_DISAGREEMENT_FACTOR`` times that minimum. The code groups by recorded
age, not a date or form identifier. Current source pairs are same-visit WG/WS
administrations; the rule does not itself verify that provenance.

The audit found counts of 385 versus 11 and 406 versus 50 at 23 months.
The external benchmark and paired WG profiles motivated retaining the lower
counts, but do not prove that the larger counts are impossible. The minimum
count guards against large ratios of small counts.

The rule is study-scoped because comparisons between other inventories need
source-specific checks. It runs after the other production masks, so its
reported catch excludes already-masked values. An implausible-production
reinstatement may still be masked here; use both flags for the combined arm.

Rows remain, with only spoken and produced masked.
``include_same_day_disagreements=True`` reinstates the values. See
``notes/202608311830-steep-within-child-gains.md``.
"""


def mask_same_day_production_disagreements(
    df: pd.DataFrame,
    *,
    include_disagreements: bool = False,
    subject_col: str = "subject_id",
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Mask production counts contradicted by a same-day count on another form.

    Applies the signature documented on :data:`SAME_DAY_DISAGREEMENT_FACTOR`.
    Within a flagged same-day group only the counts on the larger side are
    masked (``spoken`` and ``produced``); the smaller count and both rows are
    retained. The returned counts report how many observed values were masked
    per study, for the fit log. Pass ``include_disagreements=True`` to
    reintroduce them as a sensitivity.
    """
    required = {"study", "age", "spoken", subject_col}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(
            "Same-day disagreement masking requires columns: "
            + ", ".join(sorted(missing))
        )

    out = df.copy()
    dropped: dict[str, int] = {}
    if include_disagreements:
        return out, dropped

    spoken = pd.to_numeric(out["spoken"], errors="coerce")
    age = pd.to_numeric(out["age"], errors="coerce")
    observed = (
        out["study"].isin(SAME_DAY_DISAGREEMENT_STUDIES)
        & spoken.notna()
        & age.notna()
    )
    key = (
        out["study"].astype(str)
        + "::"
        + out[subject_col].astype(str)
        + "@"
        + age.astype(str)
    )
    group_min = spoken.where(observed).groupby(key).transform("min")
    group_size = observed.groupby(key).transform("sum")
    suspect = (
        observed
        & (group_size >= 2)
        & (spoken > group_min)
        & (spoken >= SAME_DAY_DISAGREEMENT_MIN_VALUE)
        & (spoken >= SAME_DAY_DISAGREEMENT_FACTOR * group_min)
    )
    if not suspect.any():
        return out, dropped

    outcome_columns = [
        column for column in ("spoken", "produced") if column in out.columns
    ]
    counts = (
        out.loc[suspect, outcome_columns]
        .notna()
        .sum(axis=1)
        .groupby(out.loc[suspect, "study"])
        .sum()
    )
    dropped = {str(study): int(count) for study, count in counts.items()}
    out.loc[suspect, outcome_columns] = float("nan")
    return out, dropped


def validate_subject_ids(
    df: pd.DataFrame,
    *,
    subject_col: str = "subject_id",
) -> None:
    """Require a non-missing, non-blank subject identifier on every row.

    Repeated-measures models namespace identifiers by study. Converting missing
    identifiers to strings would otherwise merge unrelated rows into a single
    synthetic ``"nan"`` subject and silently invalidate the clustering.
    """
    if subject_col not in df.columns:
        raise KeyError(f"Subject clustering requires column: {subject_col}")

    subject_ids = df[subject_col]
    blank = subject_ids.astype("string").str.strip().eq("").fillna(False)
    invalid = subject_ids.isna() | blank
    if invalid.any():
        raise ValueError(
            "Subject clustering requires a non-missing subject ID for every "
            f"analysis row; found {int(invalid.sum())} invalid row(s)."
        )


def exclude_us01_spoken_ceiling_rows(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, int]:
    """Drop us_01 Words & Sentences observations at its 680-word ceiling.

    This is a sensitivity-analysis transformation, not a primary inclusion
    rule. It isolates the 18 potentially right-censored Edgin WS observations
    identified in the 2026-07 export. All Words & Gestures observations,
    including a valid count at its separate 396-word ceiling, remain present.
    """
    required = {"study", "spoken", "survey_vocab_max"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(
            "us_01 ceiling sensitivity requires columns: "
            + ", ".join(sorted(missing))
        )

    at_ws_ceiling = (
        df["study"].eq("us_01")
        & df["spoken"].notna()
        & df["survey_vocab_max"].eq(US01_WS_VOCAB_MAX)
        & df["spoken"].eq(US01_WS_VOCAB_MAX)
    )
    return df.loc[~at_ws_ceiling].reset_index(drop=True), int(at_ws_ceiling.sum())


DSE_NATIVE_VOCAB_MAX: int = 810
"""The DSE Checklists' own item count, and the pool's common reference inventory.

Every model scores raw counts against ``n_trials = 810``, so for sources whose
form is *not* the DSE Checklists this is a harmonisation: a 416-item Oxford CDI
count of 200 and an 810-item DSE count of 200 are treated as the same quantity.
That is defensible only if the shorter form's items are the easier ones -- the
difficulty-ordering assumption -- which no aggregate analysis of these data can
test (see notes/202607261540 on sufficiency). Restricting the pool to rows
recorded natively at 810 removes the assumption instead of testing it, which is
what :func:`restrict_to_dse_native_administrations` is for.
"""


def restrict_to_dse_native_administrations(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, int]:
    """Keep administrations with a recorded 810-item DSE form ceiling.

    This sensitivity restricts measurement provenance rather than testing whether
    shorter forms are nested. It also changes the studies and children represented.
    The retained sources are ie_01's full follow-up wave, uk_02's DSE form and
    uk_06. ie_02 is a 476-item short form and is excluded; see
    ``DSE_SHORT_FORM_CEILINGS``. Rows with an unknown ceiling are excluded.
    """
    required = {"survey_vocab_max"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(
            "The DSE-native sensitivity requires columns: "
            + ", ".join(sorted(missing))
        )

    native = pd.to_numeric(df["survey_vocab_max"], errors="coerce").eq(
        DSE_NATIVE_VOCAB_MAX
    )
    return df.loc[native].reset_index(drop=True), int((~native).sum())


FORM_AGE_FLOORS: dict[str, dict[int, int]] = {
    "us_01": {396: 8, 680: 16},
}
"""Study-specific lower admission ages, keyed by native form ceiling.

``us_01`` now uses item-level contributor files that include records outside
Wordbank's registered norming windows. The study retains administrations above
the windows because an early-vocabulary form can be appropriate for older DS
children. Those visits also supply comprehension absent from WS forms.

The lower floor is a separate source-handling decision. The below-floor block
contains very high infant speech counts and large recorded comprehension
changes. Misrecorded ages are one possible cause; aggregate totals do not prove
it. ``include_below_form_floor=True`` reinstates these records for sensitivity.

Age alone does not identify suspect older records. Child-level ceiling concerns
use ``CEILING_ONLY_CHILD_STUDIES`` instead. See ``data/vocab_data_us_01.md``.
"""


CEILING_ONLY_CHILD_STUDIES = ("us_01",)
"""Studies with a documented concern about children recorded only near ceilings.

The Edgin audit identified batches of ceiling-level records with no alternative
count from the same child. This study-scoped rule removes children whose every
raw spoken count is at least ``IMPLAUSIBLE_PRODUCTION_CEILING_FRACTION`` of its
form ceiling. A missing count or ceiling prevents that row meeting the test.
The rule runs before masking, so it does not depend on earlier outcome masks.

This is an outcome-dependent exclusion. It can remove genuine high-vocabulary
children and usable comprehension values as well as suspect production counts.
Neither older age nor a ceiling count establishes a preparation error.
``include_ceiling_only_children=True`` tests dependence on the exclusion; it
does not establish which source records are defective.

See ``notes/202607261245-edgin-duplicated-outcome-records.md`` section 13 and
``data/vocab_data_us_01.md`` for the source history and recorded losses.
"""


def exclude_below_form_floor(
    df: pd.DataFrame,
    *,
    include_below_floor: bool = False,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Drop administrations given below their form's lowest registered age.

    Applies only to the studies in :data:`FORM_AGE_FLOORS`. Rows whose study is not
    listed, or whose ``survey_vocab_max`` does not identify a known form, are kept --
    the rule never guesses a floor it does not have. Administrations *above* a form's
    window are deliberately untouched; read :data:`FORM_AGE_FLOORS` for why.

    Returns the filtered frame and the number of rows dropped per study.
    """
    required = {"study", "age", "survey_vocab_max"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(
            "Form-floor exclusion requires columns: " + ", ".join(sorted(missing))
        )

    if include_below_floor:
        return df.reset_index(drop=True), {}

    age = pd.to_numeric(df["age"], errors="coerce")
    ceiling = pd.to_numeric(df["survey_vocab_max"], errors="coerce")
    drop = pd.Series(False, index=df.index)

    for study, floors in FORM_AGE_FLOORS.items():
        in_study = df["study"].eq(study)
        for form_ceiling, age_min in floors.items():
            drop |= in_study & ceiling.eq(form_ceiling) & age.notna() & (age < age_min)

    dropped = {
        str(study): int(count)
        for study, count in df.loc[drop, "study"].value_counts().items()
    }
    return df.loc[~drop].reset_index(drop=True), dropped


def exclude_ceiling_only_children(
    df: pd.DataFrame,
    *,
    include_ceiling_only: bool = False,
    subject_col: str = "subject_id",
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Drop every administration of a child recorded only at the form ceiling.

    Scoped to :data:`CEILING_ONLY_CHILD_STUDIES`. A child with at least one production
    count below :data:`IMPLAUSIBLE_PRODUCTION_CEILING_FRACTION` of its form's ceiling is
    kept in full -- the criterion identifies children whose *entire* record is
    ceiling-saturated, which is the documented batch signature, not individual extreme
    counts. Applied to raw source counts before any masking, so it does not depend on
    the order the other rules run in.

    Returns the filtered frame and the number of rows dropped per study.
    """
    required = {"study", "spoken", "survey_vocab_max", subject_col}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(
            "Ceiling-only exclusion requires columns: " + ", ".join(sorted(missing))
        )

    if include_ceiling_only:
        return df.reset_index(drop=True), {}

    spoken = pd.to_numeric(df["spoken"], errors="coerce")
    ceiling = pd.to_numeric(df["survey_vocab_max"], errors="coerce")
    at_ceiling = (
        spoken.notna()
        & ceiling.notna()
        & (spoken >= IMPLAUSIBLE_PRODUCTION_CEILING_FRACTION * ceiling)
    )

    in_scope = df["study"].isin(CEILING_ONLY_CHILD_STUDIES)
    # A child is identified by study and subject, so a shared subject label in two
    # studies cannot merge them.
    key = df["study"].astype(str) + "::" + df[subject_col].astype(str)
    all_at_ceiling = at_ceiling.groupby(key).transform("all")

    drop = in_scope & all_at_ceiling
    dropped = {
        str(study): int(count)
        for study, count in df.loc[drop, "study"].value_counts().items()
    }
    return df.loc[~drop].reset_index(drop=True), dropped


def select_one_observation_per_subject(
    df: pd.DataFrame,
    *,
    random_seed: int,
    study_col: str = "study",
    subject_col: str = "subject_id",
) -> pd.DataFrame:
    """Retain one reproducibly sampled administration per study-specific child.

    Random selection avoids systematically retaining the earliest or latest
    assessment. The original row order is restored after sampling so downstream
    coding and diagnostics remain deterministic.
    """
    required = {study_col, subject_col}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(
            "Single-administration selection requires columns: "
            + ", ".join(sorted(missing))
        )
    if not df.index.is_unique:
        raise ValueError(
            "Single-administration selection requires a unique dataframe index."
        )
    validate_subject_ids(df, subject_col=subject_col)

    shuffled = df.sample(frac=1.0, random_state=random_seed)
    selected = shuffled.drop_duplicates([study_col, subject_col], keep="first")
    return selected.sort_index().reset_index(drop=True)


def filter_studies_by_min_obs(
    df: pd.DataFrame,
    min_obs: int | None,
    study_col: str = "study",
) -> tuple[pd.DataFrame, list[str]]:
    """Drop studies (datasets) with fewer than ``min_obs`` observations.

    Used by the study-random-intercept models (e.g. VG11/VG12/VG13) to trim
    tiny studies that would otherwise each add a near-unidentified intercept
    without materially informing the estimates.

    Parameters
    ----------
    df
        Analysis frame (one row per observation), already filtered to the rows
        that inform the model.
    min_obs
        Minimum observations a study must have to be kept. ``None`` or ``0``
        keeps every study.
    study_col
        Column identifying the study/dataset grouping.

    Returns
    -------
    tuple[pandas.DataFrame, list[str]]
        The filtered frame (index reset) and the sorted list of dropped study
        labels.
    """
    if not min_obs:
        return df.reset_index(drop=True), []
    sizes = df.groupby(study_col).size()
    keep = sizes[sizes >= min_obs].index
    dropped = sorted(set(sizes.index) - set(keep))
    filtered = df[df[study_col].isin(keep)].reset_index(drop=True)
    return filtered, dropped


def _sql_string_list(values: tuple[str, ...]) -> str:
    """Render a tuple of strings as a quoted SQL ``IN``-list body."""
    return ", ".join(f"'{v}'" for v in values)


# Native checklist ceilings (``survey_vocab_max``), by source form (issue #128):
#   - DSE Checklists (1+2+3 = 120+340+350) = 810 words. This is the common
#     reference inventory every model's likelihood scores counts against
#     (``n_trials = 810``), so DSE-native studies (uk_02 DSE form, ie_01's
#     follow-up wave, uk_06) carry survey_vocab_max = 810.
#   - DSE Checklists 1 + 2 only (ie_02) = 476 achievable words; Checklist 3 was
#     not administered. Read from the source, and kept on the 810 scale as a
#     short form rather than masked as partial (see DSE_SHORT_FORM_CEILINGS).
#     ie_01's baseline wave is the same instrument subset recorded as zeros on
#     Checklist 3, carried as 460 and masked (INCOMPLETE_ADMINISTRATION_CEILINGS).
#   - Oxford CDI = 416 words (uk_02 Oxford form, uk_03, uk_04, uk_05).
#   - MacArthur-Bates CDI: Words & Gestures (WG) = 396 (us_01 WG form, us_02 —
#     which carries comprehension, so it is the WG form); Words & Sentences
#     (WS, production only) = 680 (us_01 WS form).
#   - NZCDI (nz_01) = 675.
#   - CDI-Down (es_01) = 651 words, the Spanish MB-CDI adaptation for children with
#     Down syndrome. Two of its children sit exactly at the comprehension ceiling
#     (legitimate but censored: their true receptive vocabulary is at least 651),
#     which the guard keeps — it drops only counts strictly above the ceiling.
#   - Reading CDI (uk_07) = 674 words, the University of Reading adaptation used by
#     the PACT-DS trial, which adds a per-item sign coding. Nothing in the source
#     reaches the ceiling on any of the four counts.
#   uk_01, it_01 and uk_07 carry a per-row source ceiling.
#
# Form-ceiling guard (issues #128/#131): exclude rows whose word count exceeds
# the native item ceiling of the checklist form they came from
# (``survey_vocab_max``). Such counts are impossible — a data-entry error, e.g.
# an it_01 row recording 461 words understood on a 408-item form — and must not
# reach any model. Rows with an unknown ceiling (``survey_vocab_max`` NULL) are
# kept, as are counts at the ceiling (a legitimate ceiling observation); only a
# count strictly above its form's ceiling is dropped.
_CEILING_GUARD_KEEP = (
    "survey_vocab_max IS NULL OR ("
    "(understood IS NULL OR understood <= survey_vocab_max) AND "
    "(spoken IS NULL OR spoken <= survey_vocab_max) AND "
    "(signed IS NULL OR signed <= survey_vocab_max) AND "
    "(produced IS NULL OR produced <= survey_vocab_max))"
)


def vocab_combined_view_sql() -> str:
    """Return the ``CREATE VIEW vocab_combined`` statement.

    The view unions the per-study tables built by ``scripts/prepare_data.py``
    into the DS analysis relation read by :func:`load_combined_data`.
    The comprehension-form guard matches the TD guard in :func:`load_data`.
    ``tests/test_data_utils.py`` checks the per-study transformations. Source
    records in ``data/vocab_data_<study>.md`` explain measurement conventions.

    The DS (Edgin) subset comes from ``vocab_us_01``, which is derived from the
    English (American) item-level contributor files and so is English by
    construction. It no longer needs the :data:`ENGLISH_LANGUAGES` filter the
    ``wordbank_child`` export required. That constant still scopes the TD loader.
    """
    bivariate_forms_sql_list = _sql_string_list(WORDBANK_BIVARIATE_FORMS)
    return f"""
    CREATE VIEW vocab_combined AS
    SELECT * FROM (
    SELECT 'uk_01' as study,
           vuk1.subject_id,
           CASE vuk1.sex WHEN 1 THEN 'M' WHEN 2 THEN 'F' END as sex,
           vuk1.age,
           vuk1.understood,
           vuk1.spoken,
           vuk1.signed,
           vuk1.produced,
           vuk1.survey_vocab_max
    FROM vocab_uk_01 as vuk1
    UNION ALL
    SELECT 'uk_02'          as study,
           vuk2.subject_id,
           CASE vuk2.sex WHEN 1 THEN 'M' WHEN 2 THEN 'F' END as sex,
           vuk2.age age,
           vuk2.comprehension as understood,
           vuk2.spoken,
           vuk2.signed,
           vuk2.production as produced,
           CASE
               WHEN vuk2.form = 'DSE' THEN 810
               WHEN vuk2.form = 'Oxford_CDI' THEN 416
               ELSE NULL
           END                as survey_vocab_max
    FROM vocab_uk_02 as vuk2
    UNION ALL
    SELECT 'ie_01'                                                   as study,
           vie.subject_id,
           NULL                                                        as sex,
           vie.age_months_start                                        as age,
           vie.understands_total_start                                 as understood,
           vie.says_total_start                                        as spoken,
           null                                                        as signed,
           null                                                        as produced,
           460                                                         as survey_vocab_max
    FROM vocab_ie_01 as vie
    UNION ALL
    SELECT 'ie_01'                                               as study,
           vie.subject_id,
           NULL                                                    as sex,
           vie.age_months_end                                      as age,
           vie.understands_total_end                               as understood,
           vie.says_total_end                                      as spoken,
           null                                                    as signed,
           vie.says_total_end                                      as produced,
           810                                                     as survey_vocab_max
    FROM vocab_ie_01 as vie
    UNION ALL
    SELECT 'us_01'                                     as study,
           concat('id_', hex(hash(vus01.subject_id)))   as subject_id,
           vus01.sex,
           vus01.age,
           CASE
               WHEN vus01.form IN ({bivariate_forms_sql_list}) THEN vus01.comprehension
               ELSE NULL
           END                                          as understood,
           vus01.production                             as spoken,
           null                                         as signed,
           vus01.production                             as produced,
           vus01.survey_vocab_max
    FROM vocab_us_01 as vus01
    WHERE vus01.dev_status = 'down_syndrome'
    UNION ALL
    SELECT 'us_03'                          as study,
           vus03.subject_id,
           NULL                                as sex,
           vus03.age,
           vus03.understood,
           NULL                                as spoken,
           NULL                                as signed,
           vus03.produced,
           vus03.survey_vocab_max
    FROM vocab_us_03 as vus03
    UNION ALL
    SELECT 'uk_03'                           as study,
           vuk2025.subject_id,
           NULL                                as sex,
           vuk2025.age,
           vuk2025.comprehension               as understood,
           vuk2025.production                  as spoken,
           null                                as signed,
           vuk2025.production                  as produced,
           416                                 as survey_vocab_max
    FROM vocab_uk_03 as vuk2025
    UNION ALL
    SELECT 'it_01'                           as study,
           vit2013.subject_id,
           NULL                                as sex,
           vit2013.age,
           vit2013.understood,
           vit2013.spoken,
           null                                as signed,
           vit2013.spoken                      as produced,
           vit2013.form_max_spoken             as survey_vocab_max
    FROM vocab_it_01 as vit2013
    UNION ALL
    SELECT 'uk_04'                           as study,
        vuk2013.subject_id,
        NULL                                as sex,
        vuk2013.age,
        vuk2013.understood,
        vuk2013.spoken,
        vuk2013.signed,
        vuk2013.spoken                      as produced,
        416                                 as survey_vocab_max
    FROM vocab_uk_04 as vuk2013
        UNION ALL
    SELECT 'uk_05'                           as study,
        vuk05.subject_id,
        CASE vuk05.sex WHEN 1 THEN 'M' WHEN 2 THEN 'F' END as sex,
        vuk05.age,
        vuk05.understood,
        vuk05.spoken,
        vuk05.signed,
        vuk05.spoken                      as produced,
        416                                 as survey_vocab_max
    FROM vocab_uk_05 as vuk05
        UNION ALL
    SELECT 'us_02'                           as study,
        vus02.subject_id,
        NULL                                as sex,
        vus02.age,
        vus02.understood,
        vus02.spoken,
        NULL                                as signed,
        vus02.spoken                     as produced,
        396                                 as survey_vocab_max  -- MacArthur-Bates WG
    FROM vocab_us_02 as vus02
        UNION ALL
    SELECT 'uk_06'                           as study,
        vuk06.subject_id,
        CASE vuk06.sex WHEN 1 THEN 'M' WHEN 2 THEN 'F' END as sex,
        vuk06.age,
        vuk06.understood,
        vuk06.spoken,
        vuk06.signed                                as signed,
        vuk06.spoken                      as produced,
        810                                 as survey_vocab_max
    FROM vocab_uk_06 as vuk06
        UNION ALL
    SELECT 'ie_02'                           as study,
        vie2.subject_id,
        CASE vie2.sex WHEN 1 THEN 'M' WHEN 2 THEN 'F' END as sex,
        vie2.age,
        vie2.understood,
        vie2.spoken,
        vie2.signed                         as signed,
        vie2.spoken                         as produced,
        vie2.survey_vocab_max               as survey_vocab_max
    FROM vocab_ie_02 as vie2
    WHERE vie2.english_speaking = 'yes'
    UNION ALL
    SELECT 'nz_01'                                        as study,
        vnz01.subject_id,
        NULL                                              as sex,
        vnz01.age,
        NULL                                              as understood,
        vnz01.spoken + vnz01.spoken_signed                as spoken,
        vnz01.signed + vnz01.spoken_signed                as signed,
        vnz01.spoken + vnz01.signed + vnz01.spoken_signed as produced,
        675                                               as survey_vocab_max
    FROM vocab_nz_01 as vnz01
    UNION ALL
    SELECT 'es_01'                           as study,
        ves01.subject_id,
        CASE ves01.sex WHEN 1 THEN 'M' WHEN 2 THEN 'F' END as sex,
        ves01.age,
        ves01.understood,
        ves01.spoken,
        CASE
            WHEN ves01.gestured <= ves01.spoken_or_gestured THEN ves01.gestured
            ELSE NULL
        END                                 as signed,
        ves01.spoken_or_gestured            as produced,
        651                                 as survey_vocab_max  -- CDI-Down
    FROM vocab_es_01 as ves01
    WHERE ves01."group" = 'DS'
    UNION ALL
    SELECT 'uk_07'                                        as study,
        vuk07.subject_id,
        CASE vuk07.sex WHEN 1 THEN 'M' WHEN 2 THEN 'F' END as sex,
        vuk07.age,
        vuk07.understood,
        vuk07.spoken + vuk07.spoken_signed                as spoken,
        vuk07.signed + vuk07.spoken_signed                as signed,
        vuk07.produced,
        vuk07.survey_vocab_max  -- 674, constant, carried in the source
    FROM vocab_uk_07 as vuk07
    ) vc
    WHERE {_CEILING_GUARD_KEEP}
    """


def _deterministic_row_order(df: pd.DataFrame) -> pd.DataFrame:
    """Return a canonical row order independent of database scan order.

    The prepared-frame hash includes row order, so reloads must use the same
    order. Sort on every column with missing values last. Exact duplicate rows
    can exchange positions without changing the hash. Call before masking.
    """
    return df.sort_values(
        list(df.columns), kind="stable", na_position="last"
    ).reset_index(drop=True)


def load_combined_data(
    max_age_months=None,
    *,
    include_incomplete_administrations=False,
    include_duplicated_outcomes=False,
    include_implausible_production=False,
    include_below_form_floor=False,
    include_ceiling_only_children=False,
    include_comprehension_below_production=False,
    include_same_day_disagreements=False,
    include_structurally_distinct_subsamples=False,
    mask_dse_short_form_comprehension=False,
    include_produced=False,
):
    """
    Load the combined data from the DuckDB database.

    (Run ./scripts/prepare_data.py to create the database if it doesn't exist.)

    Counts from partial administrations are masked by default
    (:func:`mask_incomplete_administrations`), because they are not on the
    810-item reference scale the model likelihoods assume. This is applied here
    for all DS consumers. Signing-source masking remains in the signing engines.

    Parameters:
    -----------
        max_age_months (int, optional): The maximum age in months to include in the data. Defaults to None (no limit).
        include_incomplete_administrations (bool): Reintroduce counts from partial
            administrations, for sensitivity analysis. Defaults to False.
        include_duplicated_outcomes (bool): Reintroduce infant administrations whose
            outcome columns appear duplicated, for sensitivity analysis. Defaults
            to False.
        include_implausible_production (bool): Reintroduce production counts matching
            the near-ceiling or longitudinal-collapse signature, for sensitivity
            analysis. Defaults to False.
        include_below_form_floor (bool): Reintroduce administrations given below their
            form's lowest registered age, for sensitivity analysis. Defaults to False.
            See :data:`FORM_AGE_FLOORS` for the source-specific rationale.
        include_ceiling_only_children (bool): Reintroduce children whose every
            administration is near its form ceiling, for sensitivity analysis.
            Defaults to False. See :data:`CEILING_ONLY_CHILD_STUDIES` for the
            outcome-dependent exclusion and its limits.
        include_comprehension_below_production (bool): Reintroduce comprehension
            counts below ``max(produced, spoken)``, for
            sensitivity analysis. Defaults to False.
        include_same_day_disagreements (bool): Reintroduce production counts
            contradicted by a same-day administration on another form, for
            sensitivity analysis. Defaults to False. Read
            :data:`SAME_DAY_DISAGREEMENT_FACTOR` for scope and thresholds.
        mask_dse_short_form_comprehension (bool): Mask the comprehension counts
            of the DSE short forms kept on the 810 scale
            (:data:`DSE_SHORT_FORM_CEILINGS`), for sensitivity analysis. Defaults
            to False, which keeps them.
        include_produced (bool): Keep the ``produced`` column in the returned
            frame. Defaults to False, preserving the historical column set every
            existing caller expects. The exploratory produced-outcome models
            (VG17/VG18) are the intended consumers.

    Returns:
    --------
        pd.DataFrame: The combined data as a DataFrame.
    """
    age_limit = max_age_months if max_age_months is not None else 1200

    with duckdb.connect(VOCABULARY_DATA_PATH, read_only=True) as con:
        df = con.execute(
            """
            SELECT
                study,
                subject_id,
                sex,
                age,
                understood,
                spoken,
                signed,
                produced,
                survey_vocab_max
            FROM vocab_combined
            WHERE age <= $1
            """,
            [age_limit],
        ).df()
    df = _deterministic_row_order(df)

    # Child-level ceiling screening needs raw counts. Apply source-admission
    # rules before within-child comparisons so excluded rows cannot affect them.
    df, _ = exclude_ceiling_only_children(
        df, include_ceiling_only=include_ceiling_only_children
    )
    df, _ = exclude_below_form_floor(df, include_below_floor=include_below_form_floor)
    # De-duplicate next, so a repeated row cannot affect the within-child
    # comparisons the later rules make.
    df, _ = drop_duplicate_administrations(df)
    df, _ = mask_incomplete_administrations(
        df, include_incomplete=include_incomplete_administrations
    )
    # Beside the other reference-scale rule, and before the comprehension-below-
    # production rule at the end, so a count this sensitivity masks is not also
    # reported as masked there.
    df, _ = mask_short_form_comprehension(
        df, mask_short_form=mask_dse_short_form_comprehension
    )
    df, _ = mask_duplicated_outcome_administrations(
        df, include_duplicated=include_duplicated_outcomes
    )
    df, _ = mask_implausible_production_administrations(
        df, include_implausible=include_implausible_production
    )
    # Report same-day masks only for values surviving earlier production rules.
    df, _ = mask_same_day_production_disagreements(
        df, include_disagreements=include_same_day_disagreements
    )
    # Before the row-local rule below: this removes whole administrations, so
    # running it first keeps the later rules' counts honest -- a row that is not
    # in the pool should not also be reported as masked.
    #
    # There is deliberately no companion rule for counts above their own form's
    # ceiling. The form-ceiling guard in `_CEILING_GUARD_KEEP` (issues #128/#131)
    # already drops those in the view, before any loader rule sees them, which is
    # why us_03's three over-ceiling observations never reach here. That guard
    # has no reinstatement flag, unlike every rule in this module; whether it
    # should is a live question (#289 task 0.1) and a larger change than an
    # ingest, because it applies to every source and all four count columns.
    df, _ = drop_structurally_distinct_subsamples(
        df,
        include_structurally_distinct_subsamples=(
            include_structurally_distinct_subsamples
        ),
    )
    # Count only comprehension values surviving earlier masks.
    df, _ = mask_comprehension_below_production(
        df, include_below_production=include_comprehension_below_production
    )
    # By default `produced` exists only for the rule above: no registered model
    # consumes it, and every caller of this function expects the historical
    # column set. The exploratory produced-outcome models opt in instead of
    # bypassing the loader (issue #266 finding 6).
    if include_produced:
        return df
    return df.drop(columns=["produced"])


def count_reinstated_implausible_production(
    max_age_months: int | None = None,
    *,
    include_same_day_disagreements: bool = False,
) -> int:
    """Count net spoken observations restored by the implausibility override.

    Difference two loader paths with the same-day flag held fixed. With that
    rule active, some reinstated values can be masked again. With both overrides
    active, the difference describes this rule's contribution to the combined arm.
    """
    masked = load_combined_data(
        max_age_months=max_age_months,
        include_same_day_disagreements=include_same_day_disagreements,
    )
    reinstated = load_combined_data(
        max_age_months=max_age_months,
        include_implausible_production=True,
        include_same_day_disagreements=include_same_day_disagreements,
    )
    return int(
        reinstated["spoken"].notna().sum() - masked["spoken"].notna().sum()
    )


def count_reinstated_same_day_disagreements(
    max_age_months: int | None = None,
    *,
    include_implausible_production: bool = False,
) -> int:
    """Count net spoken observations restored by the same-day override.

    Difference two loader paths with ``include_implausible_production`` held
    fixed. Lifting implausibility masks can expose more paired observations
    to the same-day rule.
    """
    masked = load_combined_data(
        max_age_months=max_age_months,
        include_implausible_production=include_implausible_production,
    )
    reinstated = load_combined_data(
        max_age_months=max_age_months,
        include_implausible_production=include_implausible_production,
        include_same_day_disagreements=True,
    )
    return int(
        reinstated["spoken"].notna().sum() - masked["spoken"].notna().sum()
    )


def count_masked_dse_short_form_comprehension(
    max_age_months: int | None = None,
    *,
    include_implausible_production: bool = False,
    include_same_day_disagreements: bool = False,
) -> int:
    """Comprehension observations the short-form sensitivity masks.

    The fit-log figure for ``mask_dse_short_form_comprehension``, for the same
    reason the reinstatement counts above exist: a sensitivity whose flag had
    stopped biting would otherwise look like a pass. Differenced through the
    loader with the other two engine-forwarded flags held at the definition's
    values on both sides.
    """
    kept = load_combined_data(
        max_age_months=max_age_months,
        include_implausible_production=include_implausible_production,
        include_same_day_disagreements=include_same_day_disagreements,
    )
    masked = load_combined_data(
        max_age_months=max_age_months,
        include_implausible_production=include_implausible_production,
        include_same_day_disagreements=include_same_day_disagreements,
        mask_dse_short_form_comprehension=True,
    )
    return int(
        kept["understood"].notna().sum() - masked["understood"].notna().sum()
    )


def _subsample_subjects(
    df: pd.DataFrame, sample_fraction: float, random_seed: int
) -> pd.DataFrame:
    """Sample children and retain all their administrations.

    Row sampling reduces the repeat observations needed to separate child effects
    from observation-level dispersion. See the failed row-sampled VG11 fit in
    ``notes/202608020829-kappa-and-eta-q-prior-recalibration.md`` sections 11-12.

    Keys combine study and subject identifiers. Sort keys before random sampling
    so selected children do not depend on database scan order.
    """
    subject_key = (
        df["study"].astype(str) + "::" + df["subject_id"].astype(str)
    )
    keep = (
        pd.Series(sorted(subject_key.unique()))
        .sample(frac=sample_fraction, random_state=random_seed)
    )
    return (
        df[subject_key.isin(set(keep))]
        .reset_index(drop=True)
    )


def load_data(
    population: Population,
    columns: list[str],
    sample_fraction: float = 1.0,
    random_seed: int = 47,
    max_age_months: int | None = None,
    languages: tuple[str, ...] | None = ENGLISH_LANGUAGES,
    *,
    include_incomplete_administrations: bool = False,
    include_duplicated_outcomes: bool = False,
    include_implausible_production: bool = False,
    include_below_form_floor: bool = False,
    include_ceiling_only_children: bool = False,
    include_comprehension_below_production: bool = False,
    include_same_day_disagreements: bool = False,
    include_structurally_distinct_subsamples: bool = False,
    mask_dse_short_form_comprehension: bool = False,
) -> pd.DataFrame:
    """
    Load vocabulary data for the specified population.

    Parameters
    ----------
    population : Population
        Which population to load data for.
    columns : list[str]
        Columns to select (e.g. ["age", "spoken"] or ["age", "understood", "spoken"]).
        For DS the ``study``, ``subject_id`` and ``sex`` columns are available.
        For TD the ``study`` column is aliased from ``dataset_name`` (the Wordbank
        dataset/lab identifier), along with ``subject_id``, ``form``, ``language``
        and ``sex``, recoded to the DS pool's ``'M'``/``'F'``.
    sample_fraction : float
        Fraction of **subjects** to subsample (TD only). 1.0 = no subsampling.
        Whole children are drawn and all their administrations kept, so
        within-child replication survives the subsample. See
        :func:`_subsample_subjects` for why drawing rows instead is unsafe for
        any model carrying subject random effects.
    random_seed : int
        Random seed for subsampling.
    max_age_months : int | None
        Upper age bound in months, inclusive. None uses the TD pool's default
        upper bound; for DS it uses :func:`load_combined_data`'s broad limit.
    languages : tuple[str, ...] | None
        Wordbank ``language`` values to include (TD only). Defaults to
        :data:`ENGLISH_LANGUAGES`. Pass a wider tuple to broaden the scope, or
        ``None`` to include all languages. Ignored for DS, whose studies include
        English, Italian and Spanish sources.
    include_incomplete_administrations, include_duplicated_outcomes, include_implausible_production, include_below_form_floor, include_ceiling_only_children, include_comprehension_below_production, include_same_day_disagreements : bool
        Reinstate records that :func:`load_combined_data` masks or drops by default,
        for sensitivity analysis. These flags apply to DS only; passing one for TD is a
        caller error rather than a silent no-op.
    mask_dse_short_form_comprehension : bool
        Mask the DSE short forms' comprehension counts, for sensitivity analysis
        (:func:`mask_short_form_comprehension`). **DS only**, refused for TD
        the same way.

    Returns
    -------
    pd.DataFrame
        DataFrame with the requested columns.
    """
    reinstatements = {
        "include_incomplete_administrations": include_incomplete_administrations,
        "include_duplicated_outcomes": include_duplicated_outcomes,
        "include_implausible_production": include_implausible_production,
        "include_below_form_floor": include_below_form_floor,
        "include_ceiling_only_children": include_ceiling_only_children,
        "include_comprehension_below_production": (
            include_comprehension_below_production
        ),
        "include_same_day_disagreements": include_same_day_disagreements,
        "include_structurally_distinct_subsamples": include_structurally_distinct_subsamples,
    }
    if population == Population.DOWN_SYNDROME:
        df = load_combined_data(
            max_age_months=max_age_months,
            mask_dse_short_form_comprehension=mask_dse_short_form_comprehension,
            **reinstatements,
        )
        return df[columns]

    ds_only_flags = {
        **reinstatements,
        "mask_dse_short_form_comprehension": mask_dse_short_form_comprehension,
    }
    if any(ds_only_flags.values()):
        raise ValueError(
            "Down syndrome pool flags apply to the Down syndrome pool only; "
            f"got {sorted(k for k, v in ds_only_flags.items() if v)} for {population}."
        )

    # Query the typically developing pool from wordbank_child directly.
    #
    # Wordbank's CDI: Words & Sentences (WS) rows contain valid production
    # counts, but their comprehension column is a production proxy. Keep WG
    # and Oxford CDI as bivariate observations, and include WS only for
    # spoken-only models.
    needs_understood = "understood" in columns
    needs_spoken = "spoken" in columns

    td_forms = list(WORDBANK_BIVARIATE_FORMS)
    if needs_spoken and not needs_understood:
        td_forms.extend(WORDBANK_SPOKEN_ONLY_FORMS)

    age_lower, default_upper = TD_POOL_AGE_MONTHS
    age_upper = max_age_months if max_age_months is not None else default_upper

    params: list = [td_forms, age_upper, age_lower]
    language_clause = ""
    if languages is not None:
        params.append(list(languages))
        language_clause = f"AND language IN ${len(params)}"

    # The export records some administrations twice, identically (28 exact
    # full-row copies in the 2026-06-15 download; 22 reached VG11's frame, 3
    # VG12's and 2 VG13's before this guard existed). A repeated row
    # double-weights its administration in every likelihood and, in the
    # random-effect models, makes a single-visit child look like a
    # repeated-measures one — the same defect `drop_duplicate_administrations`
    # removes from the DS pool. `SELECT DISTINCT *` runs on the complete
    # source row *before* the outcome projection below, and must stay there:
    # after projection, two genuinely distinct same-child, same-age
    # administrations can collide once the columns that separate them (`sex`,
    # `caregiver_education`, …) are dropped and WS comprehension is nulled, so
    # deduplicating the projected frame would delete real observations. The
    # per-model removal counts are pinned in tests/test_data_utils.py; the
    # audit is recorded in notes/202608231830-vg11-vg13-immediate-remediation.md.
    with duckdb.connect(VOCABULARY_DATA_PATH, read_only=True) as con:
        td_df = (
            con.execute(
                f"""
            WITH admissions AS (
                SELECT DISTINCT * FROM wordbank_child
                WHERE typically_developing = true
                    AND age <= $2
                    AND age >= $3
                    AND health_conditions IS NULL
                    AND dataset_name NOT IN ({_sql_string_list(TD_POOL_EXCLUDED_DATASETS)})
                    AND form IN $1
                    {language_clause}
            )
            SELECT
                form,
                language,
                dataset_name                       as study,
                concat('id_', hex(hash(child_id))) as subject_id,
                age,
                CASE
                    WHEN form IN ({_sql_string_list(WORDBANK_BIVARIATE_FORMS)}) THEN comprehension
                    ELSE NULL
                END                                as understood,
                production                         as spoken,
                typically_developing,
                health_conditions,
                CASE sex WHEN 'Male' THEN 'M' WHEN 'Female' THEN 'F' END as sex
            FROM admissions
            """,
                params,
            )
            .df()
        )
    td_df = _deterministic_row_order(td_df)

    if sample_fraction < 1.0:
        td_df = _subsample_subjects(td_df, sample_fraction, random_seed)

    return td_df[columns]
