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
"""Wordbank datasets barred from the typically-developing reference pool.

``Edgin`` supplies the ``us_01`` Down syndrome subset. Two of its 435 rows also
satisfy the typically-developing filter (``typically_developing`` true, no health
condition recorded) — it is the only clinical cohort in the export that leaks this
way, contributing 0.5% of its rows against at least 10% for every other dataset.
One of the two is a Words & Sentences record at exactly the 680-word ceiling,
inside the run of 21 consecutive ceiling records that
``notes/202607261245-edgin-duplicated-outcome-records.md`` §13 identifies as a
preparation artefact.

The cost is two rows of 15,379, so this changes no estimate. It is done because
the reference pool is what the Down syndrome exclusions are benchmarked against,
and a dataset whose preparation we have established to be defective — and which,
the source team having confirmed the original files are no longer available,
cannot be repaired at source — should not sit on both sides of that comparison.
"""

TD_POOL_AGE_MONTHS = (8, 30)
"""Age window, in months, of the typically-developing reference pool.

The upper bound was already implicit — the loader defaulted to 30 — and the lower
bound was implicit too, because every English CDI form in Wordbank starts at 8
months. Widening the pool beyond English made the lower bound matter: Italian Words &
Gestures is registered from **7** months, and five Italian administrations at 7 months
sit below the floor of the typically-developing models' GP domain
(``_TD_GP_DOMAIN_MONTHS = (8, 30)``), which ``build_utils`` rightly refuses.

Bounding the pool is the right fix rather than widening that domain: the GP domain is
shared with VG03/VG04, so widening it would make those models stale for the sake of
five observations at the least informative end of the range — where a
typically-developing child knows almost no words. Stating the window here keeps the
pool inside the GP domain whichever languages are admitted, instead of leaving that
invariant to depend on which forms happen to be in scope.

``max_age_months`` still overrides the upper bound per model (VG13 uses 18). There is
deliberately no per-model override for the lower bound: no model has wanted one, and
a model that did would be asking to sit outside its own GP domain.
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

DUPLICATED_OUTCOME_MAX_AGE_MONTHS = 18
DUPLICATED_OUTCOME_MIN_UNDERSTOOD = 100
DUPLICATED_OUTCOME_RATIO = 0.75
"""Signature of an administration whose two outcome columns collapsed onto one value.

An infant recorded as *saying* almost every word they understand has an internally
inconsistent administration: comprehension leading production is the most robust
finding in the early-vocabulary literature, and is the structural premise of the
joint models' ``p_S = p_U * q`` decomposition. Where that pattern appears in
infancy together with a substantial comprehension count, the likeliest explanation
is that one outcome column was written over the other at data preparation.

Detected as ``spoken >= DUPLICATED_OUTCOME_RATIO * understood`` with
``understood >= DUPLICATED_OUTCOME_MIN_UNDERSTOOD`` at
``age <= DUPLICATED_OUTCOME_MAX_AGE_MONTHS``. All three conditions are needed. The
same ratio at older ages is ordinary — a child who says most of what they
understand — so the rule is **age-conditioned rather than study-scoped**: of the
paired rows in the current pool matching the ratio and count conditions, those at
37 months or older are legitimate. Below 19 months it matches 8 rows, all in
``us_01``.

The ratio is set from the measured gap rather than chosen: among ``us_01`` Words &
Gestures administrations with comprehension >= 100, the ratios descend
1.00, 1.00, 0.99, 0.98, 0.94, 0.91, 0.90, 0.86 and then fall to 0.55 — a gap of
0.306, the largest in the distribution, so any cut inside it separates the cluster
identically. An earlier 0.9 threshold cut through the middle of that cluster and
missed two records; ``scripts/audit_edgin_subset.py`` recomputes the gap.

Three independent lines of evidence support masking these (see
``notes/202607261245-edgin-duplicated-outcome-records.md``):

- The pattern is rare where it can be checked against a large reference sample:
  among 2,480 typically-developing Words & Gestures administrations with
  comprehension >= 100, only 0.69% have production >= 0.9 * comprehension.
- The implied production levels — 134 to 396 words between 11 and 18 months — are
  impossible against the independent Berglund et al. (2001) Down syndrome cohort,
  which puts median spoken vocabulary near zero at 12 months and about 10 words at
  24 months. This is an external benchmark, not an in-sample one.
- Every affected child with a second administration shows an ordinary
  comprehension-production gap in that other record (ratios 0.08-0.13 against
  0.86-1.00 in the flagged one).

Deliberately *not* caught: administrations with a high comprehension count but a
normal production gap. Two such ``us_01`` records (comprehension 213 and 217 at 18
months, production 31 and 22) sit at the 48th and 50th typically-developing
percentile for comprehension and are retained, on the study owner's judgement that
they are clinically unusual but should not be excluded. They are a sensitivity
target, not a defect.
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

Withheld here rather than left to the general rule. Since 2026-08-25 the ten
comparable records in ``ie_01``, ``uk_01`` and ``it_01`` are masked by
:func:`mask_comprehension_below_production` -- previously they were retained and
flagged, and this docstring drew the contrast against that. The reason for
keeping a separate mechanism is unchanged: those ten are a known, stable property
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
"""ie_02 administrations withheld as internally contradictory.

Keyed by ``(subject_id, timepoint)``. One administration is listed: at 48
months (t2) this child records 442 words understood, 3 spoken and 301 signed,
against 111 understood, 72 spoken and 64 signed three months earlier at t1.
Read together, the two administrations assert a 331-word comprehension surge,
a 237-word signing surge and a 96% collapse in speech within the same three
months. The comprehension gain rate (110 words/month) is the largest in the
Down syndrome pool and sits beyond the typically-developing pool's own 99th
percentile for within-child comprehension gains (90 words/month), and
vocabulary does not shrink — let alone by 96% while comprehension quadruples.

The spoken collapse (72 → 3) is the "ie_02 record at 45 months" that
:data:`COLLAPSE_FACTOR`'s age scope deliberately left for separate
investigation. That investigation
(``notes/202608311830-steep-within-child-gains.md``) found the whole t2
administration anomalous, not just its spoken value: every count moves
implausibly at once, in the pattern of a checklist completed differently
between waves — the DSE checklists record "understands and signs" and "says"
as separate per-word columns, so words ticked under signing at t2 that t1
recorded as said would produce exactly this signature. Which columns are
trustworthy cannot be recovered from the aggregate counts, so the
administration is withheld whole, pending clarification with the source team,
rather than half-masked.

Applied at CSV load in ``scripts/prepare_data.py``, so the row is absent from
the ``vocab_ie_02`` table, the ``vocab_combined`` view and
``vocab_data_merged.csv`` alike — the same treatment as
:data:`UK07_WITHHELD_ADMINISTRATIONS`, and the same study-level precedent as
the ie_02 subject already excluded at load in ``prepare_data.py``. Removing
the entry from this tuple and re-running ``scripts/prepare_data.py``
reinstates it.
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


def mask_duplicated_outcome_administrations(
    df: pd.DataFrame,
    *,
    include_duplicated: bool = False,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Mask infant administrations whose outcome columns appear to be duplicates.

    Applies the signature documented on :data:`DUPLICATED_OUTCOME_RATIO`. Both
    counts are masked rather than one, because in the affected records neither
    value is defensible: the production figures are impossible against the
    independent Down syndrome cohort benchmark, and in the two cases where the
    child's other administration also disagrees on comprehension it falls by 355
    and 209 words. Which column was overwritten cannot be recovered from the
    aggregate data, so the administration is treated as unusable rather than
    half-repaired. The row is retained, so age coverage and provenance stay
    auditable.

    The returned counts report how many observed values were masked per study, for
    the fit log. Pass ``include_duplicated=True`` to reintroduce them as a
    sensitivity.

    Item-level responses would settle the mechanism definitively — a duplicated
    column appears as two identical response vectors — so this rule is stated as a
    signature with a stated false-positive rate, to be confirmed or refuted when
    the item-level data are ingested.
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


COMPREHENSION_BELOW_PRODUCTION_STUDIES: tuple[str, ...] = ("ie_01", "it_01", "uk_01")
"""Studies carrying a comprehension count below the child's own production count.

An inclusive comprehension field cannot be exceeded by production: a word the
child says is a word the child understands, so ``understood >= produced`` holds
by construction on any form where comprehension is asked inclusively. Ten
administrations violate it -- seven in ``ie_01``, two in ``uk_01``, one in
``it_01`` -- and the violations are not marginal: one ``ie_01`` child records 13
words understood against 366 spoken, and two record 0 understood against 83
spoken.

**The comprehension count is what gets masked, not the production count.** The
production figure is corroborated by two columns that agree (``spoken`` and
``signed`` sum to the recorded ``produced``), and in both studies with a
diagnosis the fault has been localised to comprehension: ``uk_01``'s
``understood`` appears to *exclude* words the child also produces, which is why
``spoken / understood`` reaches 1.95 there, and ``ie_01``'s seven rows sit in the
wave whose Checklist 1 comprehension field is already known to be unreliable
(pooled comprehension *falls* between waves while the mean understood total
rises). Masking the row wholesale would discard production counts that are not
in question.

``produced`` is the right denominator and ``spoken + signed`` is not. In the
signing studies the two columns overlap -- a child who both says and signs a word
is counted in each -- so their sum overstates distinct words produced, badly:
``uk_07`` has ``produced < spoken + signed`` on 77 of 82 rows and ``nz_01`` on
101 of 111. Reconstructing production as the sum would flag 87 administrations
instead of 10, almost all of them bimodal children penalised for double counting.

Equality is **kept**. ``understood == produced`` is a child who produces
everything they understand, which is legitimate; 45 administrations meet it, of
which 18 are ``0 == 0`` and most of the rest sit at the 396-item Words &
Gestures ceiling, where both counts are censored rather than equal. Those belong
to the ceiling and administration rules, not to this one.

Related, and deliberately not merged into this rule:
:data:`UK07_WITHHELD_ADMINISTRATIONS` withholds a single ``uk_07`` row with the
same signature (191 understood against 489 produced) at CSV load, because that
one is an open question with a reachable source team rather than a closed
property of the data. It never reaches ``vocab_combined``, so this rule never
sees it. Its docstring previously contrasted itself with ``ie_01``'s seven
"retained-and-flagged" records; as of 2026-08-25 those are masked here instead,
on the study owner's ruling.

Set ``include_comprehension_below_production=True`` to reinstate the ten
comprehension counts for sensitivity analysis.
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
#: data) sit at 62-80 months against 17-35 for all 286 other administrations --
#: a 27-month gap with nothing in it. They have no second visit, sit near the
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

    Requires a ``produced`` column. Comparing against ``spoken + signed`` instead
    is wrong wherever the two modalities overlap, so a frame without ``produced``
    raises rather than silently substituting a different rule.

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
    produced = pd.to_numeric(out["produced"], errors="coerce")
    suspect = understood.notna() & produced.notna() & (understood < produced)
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
"""Two signatures of a production count that cannot be a real measurement.

Both are scoped to ``age <= IMPLAUSIBLE_PRODUCTION_MAX_AGE_MONTHS``, the window in
which the independent Berglund et al. (2001) Down syndrome cohort puts median
spoken vocabulary near zero at 12 months and about 10 words at 24, and in which its
single most able child of 330 had not yet approached the counts in question — that
child reached 668 words at 48 months.

That age scope is load-bearing and deliberately **not** relaxed. Above it a
near-ceiling count is ordinary rather than suspect, and removing the bound would mask
19 apparently legitimate records across six studies — a uk_01 child at 115 months with
658 of 680 words, an ie_01 child at 69 months with 741 of 810, an es_01 child at 54
months with 637 of 651. Age and count together therefore cannot separate a legitimate
able older child from the Edgin ceiling batch; that batch is identified on its
provenance instead, by :data:`CEILING_ONLY_CHILD_STUDIES`, which runs first and leaves
nothing above 30 months for this rule to find.

**Near-ceiling saturation.** ``spoken >= IMPLAUSIBLE_PRODUCTION_CEILING_FRACTION *
survey_vocab_max``. Within the scoped window this now matches 8 ``us_01``
administrations, down from 21, because :func:`exclude_ceiling_only_children` runs first
and removes the batch children wholesale rather than masking their counts one at a
time. What reaches this rule are ceiling counts from children who *do* have other,
non-ceiling records. For scale, no typically-developing child of 1,469 aged 16-19
months reaches the Words & Sentences ceiling, and their maximum is 643.

The batch signature that first identified these is recorded in
``notes/202607261245-edgin-duplicated-outcome-records.md`` §13: thirteen records in a
contiguous Wordbank ``child_id`` block, every one at exactly 680, no child with any
other administration, and an 81-id gap to the next id present. Those ids are no
longer this repository's identifiers — ``us_01`` now keys on the study's own subject
id (see :data:`CEILING_ONLY_CHILD_STUDIES` and ``scripts/build_us01_source.py``), because
Wordbank issued a **separate child_id per form**, so the 119 apparent children were
53 Words & Gestures records plus 66 Words & Sentences records with no child linked
across the two. They are 71 children, 46 of whom took both forms. The note's
reasoning stands; only the identifiers it cites are historical.

**Where the ceiling records are concentrated.** Every Words & Gestures record at or
near the 396-item ceiling — 25 of them, including all six of the oldest at 61, 62,
63, 73, 84 and 173 months — sits *outside* the form's age window, as do all 62 Words
& Sentences records above 30 months, of which 61 are at exactly 680. Those are held
back by :func:`exclude_ceiling_only_children` before this rule is reached, which is
why the near-ceiling count here is lower than the whole cohort's would suggest.

**Longitudinal collapse.** A count of at least ``COLLAPSE_MIN_VALUE`` that exceeds
the same child's later count by a factor of ``COLLAPSE_FACTOR`` or more. Vocabulary
does not shrink, so this is unambiguous — 656 words at 17 months against 12 at 23
months is not measurement noise. The floor matters: without it the rule fires on
trivial pairs such as 5 understood words falling to 1. The age scope matters too:
at older ages a decline can arise from a form change or from noise in large counts,
and two such records outside ``us_01`` (a uk_01 record at 76 months, an ie_02
record at 45) were deliberately left for separate investigation rather than masked
by a rule whose justification is developmental. Both investigations have since
concluded (2026-08-31): the uk_01 record was two same-named children fused under
one name-derived id, resolved by :data:`UK01_WITHHELD_SUBJECTS`, and the ie_02
record's whole follow-up administration proved internally contradictory, resolved
by :data:`IE02_WITHHELD_ADMINISTRATIONS`. Leaving them out of this rule was the
right call in both cases — neither was a developmental collapse.

Within the scoped window the two signatures together mask 11 ``us_01`` spoken counts
and nothing in any other study. That is fewer than the 30 masked before the source
change, and the reason is not that less is caught but that
:func:`exclude_ceiling_only_children` removes the ceiling batch as whole children
first, so those counts never reach this rule. A Words & Sentences record of 406 words
at 23 months was long retained here as extreme against the external benchmark but
lacking a positive defect signature — no later administration contradicted it. Its
*same-day* Words & Gestures administration does contradict it (50 words), which is
now its own signature: :data:`SAME_DAY_DISAGREEMENT_FACTOR` masks that record and
one other. The high-comprehension records described on
:data:`DUPLICATED_OUTCOME_RATIO` remain retained sensitivity targets.
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
"""Signature of a production count contradicted by a same-day count on another form.

46 of the ``us_01`` children took Words & Gestures and Words & Sentences at the
same visit, giving two same-day measurements of the same construct — the forms
share the MacArthur core vocabulary, so two same-day production counts cannot
legitimately be far apart. The pool's same-age pairs bear this out: all but two
agree closely, and the largest disagreement among the rest is 10 against 71, at
counts where a handful of ticks moves the ratio.

The two violations are stark. At 23 months one child records 11 words spoken on
Words & Gestures against **385** on Words & Sentences the same day; another
records 50 against **406**. 385 and 406 words spoken at 23 months are
impossible against the independent Berglund et al. (2001) Down syndrome
benchmark (median spoken vocabulary near 10 words at 24 months; the single most
able child of 330 reached 668 words at 48 months). The 406 record is the one
:data:`COLLAPSE_FACTOR`'s docstring long retained as extreme-but-uncontradicted
— the same-day Words & Gestures administration *is* the contradiction, which is
why this rule now exists.

Detected within a ``(study, subject, age)`` group holding two or more observed
production counts: the group is flagged when its largest count is at least
``SAME_DAY_DISAGREEMENT_MIN_VALUE`` and at least
``SAME_DAY_DISAGREEMENT_FACTOR`` times its smallest. **Only the larger side is
masked.** The smaller count is corroborated twice over — by the external
benchmark and, in both flagged cases, by the child's independently measured
Words & Gestures comprehension-production gap (comprehension 279 and 89 against
production 11 and 50) — so discarding it would throw away a defensible
measurement. This differs from :data:`DUPLICATED_OUTCOME_RATIO`, which masks
both columns because there neither is defensible. The floor keeps small-count
noise out (10 against 71 is untouched); the factor matches
:data:`COLLAPSE_FACTOR`.

The rule runs **after** the other production rules, deliberately: several
same-day pairs in the ceiling region at 17-18 months would also match this
signature, but the near-ceiling, collapse and duplicated-outcome rules already
mask them, so running last keeps those rules' documented counts unchanged and
makes this rule's catch exactly the counts nothing else explains — within the
current pool, the two Words & Sentences counts above and nothing else. The net
masking is order-independent. One interaction follows from the overlap: under
the ``include_implausible_production`` sensitivity this rule still sees the
reinstated ceiling-region pairs and independently re-masks those with an
observed same-day partner, so that sensitivity's net reinstatement is smaller
than the implausible rule's own catch —
:func:`count_reinstated_implausible_production` reports the true figure.

Study-scoped to ``us_01`` deliberately. uk_01's six same-day WG/WS pairs agree
closely, so there is nothing to catch; uk_02's same-day pairs are Oxford (416)
against DSE (810) counts, whose inventories differ enough that a factor-two
disagreement is mechanical rather than contradictory — a cross-inventory
version of this rule would need the dual-form crosswalk, not a threshold.

Both rows are retained and only ``spoken``/``produced`` masked, so age coverage
and provenance stay auditable. Set ``include_same_day_disagreements=True`` to
reinstate the masked counts for sensitivity analysis. See
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
    """Keep only administrations recorded natively on the 810-item DSE Checklists.

    A sensitivity-analysis transformation, not a primary inclusion rule. It
    answers what the trajectories look like when no count has been carried onto
    a denominator its form did not use: 277 of the Down syndrome pool's 1,516
    rows survive, from 194 children across ie_01 (its 810 wave only), ie_02,
    uk_02 (DSE form only) and uk_06 -- 251 understood, 263 spoken and 217 signed
    observations spanning 9-115 months. (Understood was 259 before
    :func:`mask_comprehension_below_production`, whose ten masked counts fall
    seven inside this subset, all in ie_01, and 252 before the withheld ie_02
    administration -- :data:`IE02_WITHHELD_ADMINISTRATIONS` -- left the
    pool.) Every other source is on a shorter form
    and drops out entirely, es_01, nz_01, uk_07 and us_01 among them.

    Rows whose ceiling is unrecorded are dropped rather than kept: an unknown
    form cannot be shown to be the native one, and the point of the variant is
    to admit only what is known to need no harmonisation.
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
"""Lowest age, in months, at which each source form may be administered.

Keyed by study and form ceiling. Wordbank registers an ``age_min``/``age_max`` per
instrument -- English (American) Words & Gestures 8-18 months, Words & Sentences 16-30
-- and its by-child download page silently drops every administration outside that
window. ``scripts/build_us01_source.py`` reads the item-level contributor files
instead, so ``us_01`` now carries the administrations the export never showed.

**Only the floor is enforced.** Administrations *above* a form's window are admitted:
for a Down syndrome cohort, giving an early-vocabulary form to a chronologically older
child is developmentally appropriate rather than an error, and the age window governs
whether Wordbank's *percentile norms* apply -- which this project does not use. Every
model scores raw counts against the 810-item reference with a per-form ceiling guard,
and a raw count is a raw count at any age.

Excluding them would also have been the more biased choice, not the safer one. A child
still on Words & Gestures at 25 months is plausibly lower-ability than one who had
moved to Words & Sentences, so dropping the whole out-of-window block removes
observations non-randomly with respect to ability. Concretely, ``us_01`` contributes 58
administrations between 19 and 27 months and every one is Words & Sentences, whose
comprehension is a production proxy discarded by :data:`WORDBANK_BIVARIATE_FORMS` --
so before these rows were admitted the study contributed **no comprehension
observations at all** in that band. The 50 Words & Gestures administrations admitted
there are its only ones, and all 47 contributing children are already in the pool, so
they are repeat visits carrying within-child information rather than new children.

The floor is enforced because it is a different case. 16 ``us_01`` administrations sit
below their form's floor, at 5-7 months, and three of them are physically impossible:
236, 364 and 368 words *spoken*, which no 6-month-old in any population produces. Two
more of the same children show comprehension collapsing from 247-371 words at 6 months
to 5-19 by 11-12 months. The block is unreliable, most likely mis-keyed ages, and the
remaining rows in it are near-zero counts that carry almost no information anyway.

The genuinely defective out-of-window administrations are handled on their own
evidence, not by age: see :data:`CEILING_ONLY_CHILD_STUDIES`.
"""


CEILING_ONLY_CHILD_STUDIES = ("us_01",)
"""Studies in which a child recorded only at the form ceiling is a preparation artefact.

``notes/202607261245-edgin-duplicated-outcome-records.md`` §13 identified a batch
signature in the Edgin subset: a run of records at exactly the form ceiling in which no
affected child has any other administration. With the full source now ingested that
signature resolves 64 children and 98 administrations -- 23 Words & Gestures at 39-173
months, all at exactly 396 spoken; 62 Words & Sentences at 31-88 months, 61 at exactly
680; and 13 Words & Sentences at 24-30 months whose counts every other rule already
masks, so removing them changes no estimate.

**Why a provenance criterion rather than an age one.** A near-ceiling count is a defect
signature only in infancy, where the Berglund benchmark rules it out. At older ages it
is ordinary: an eight-year-old with Down syndrome knowing 658 of 680 words is expected.
Removing the age scope from :data:`IMPLAUSIBLE_PRODUCTION_MAX_AGE_MONTHS` would
therefore mask 19 apparently legitimate records across six other studies (uk_01 at 115
months with 658 of 680, ie_01 at 69 with 741 of 810, es_01 at 54 with 637 of 651, and
so on). Age and count together cannot separate the Edgin batch from those. What does
separate them is that **the batch children have no non-ceiling record of their own** --
a fact about how the data were prepared, not about the values, so it is not selection
on the outcome.

**Why it is study-scoped where the duplicated-outcome rule deliberately is not.** That
rule's evidence is developmental and so applies to any study. This one's evidence is a
specific, documented failure of one dataset's preparation, whose source team has
confirmed the original files no longer exist. Applying it elsewhere would assert a
defect for which there is no evidence.

**What it costs, stated plainly.** The rule is not free: 23 of the removed
administrations carry a live comprehension value, all at exactly 396 on the 396-item
Words & Gestures form between 39 and 173 months. A child recorded as understanding every
word *and* saying every word at 173 months is the artefact rather than a measurement, so
they go with the rest of their record — but the 23 are a real loss, not bookkeeping.

Because the criterion is applied to raw source counts before any masking, it also removes
14 children who were previously in the pool as *phantoms*: every one of their counts was
already masked by another rule, so they contributed a subject random effect informed only
by its prior. Of the 71 children the previous ``us_01`` pool reported, 57 had at least one
live observation; the figure after this rule is 58.
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
#     (``n_trials = 810``), so DSE-native studies (uk_02 DSE form, ie_01, uk_06,
#     ie_02) carry survey_vocab_max = 810.
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
    into the single DS analysis relation read by :func:`load_combined_data`.
    It is defined here rather than inline in the script so the per-study
    transformations — in particular the us_01/Edgin Wordbank form guard,
    which must stay in lockstep with the TD guard in :func:`load_data` — are
    importable and regression-tested (see ``tests/test_data_utils.py``).

    The DS (Edgin) subset comes from ``vocab_us_01``, which is derived from the
    English (American) item-level contributor files and so is English by
    construction — it no longer needs the :data:`ENGLISH_LANGUAGES` filter the
    ``wordbank_child`` export required. That constant still scopes the TD loader.
    """
    bivariate_forms_sql_list = _sql_string_list(WORDBANK_BIVARIATE_FORMS)
    return f"""
    CREATE VIEW vocab_combined AS
    SELECT * FROM (
    -- `sex` in the per-source CSVs is the canonical 1 = male / 2 = female
    -- coding (research-data-analysis prepare/readme.md, "Sex coding"). It is
    -- decoded to M/F here, which is the representation this view has always
    -- exposed and which `us_01` — built in this repo by scripts/build_us01_source.py
    -- — already produces. Before that standardisation each source arrived in its
    -- own coding and this view handled them one at a time: uk_02's 0/1 was
    -- decoded here, uk_06's boy/girl was discarded as NULL, and uk_05's sex
    -- never reached its CSV at all. us_01 is unaffected either way -- it is
    -- built in this repo, not taken from research-data-analysis.
    --
    -- ie_02 is decoded the same way, but its coding is the one resting on a
    -- confirmation rather than on the file: its source carries 1/2 with no
    -- value label saying which is which. It was carried upstream as
    -- `sex_source_code` and NULL here until the contributor confirmed, on
    -- 2026-09-04, that it is the same 1 = male / 2 = female coding as every
    -- other source; upstream then renamed the column to plain `sex`.
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
    -- ie_01 (Down Syndrome Ireland), two waves.
    --
    -- ``understood`` is the parent-reported comprehension count, passed through
    -- unchanged. It was previously GREATEST(says_total, understands_total) on the
    -- reasoning that production implies comprehension; that repaired 7 records in
    -- which says > understands by overwriting comprehension with production, which
    -- (a) hid them from the ``n_parent_violations`` count that methods-models.qmd
    -- says such rows are reported through, and (b) fed the nested spoken
    -- likelihood exact S = U rows, i.e. observations that the child says every
    -- word it understands. Repairing a count from the outcome being modelled is
    -- selection on the outcome; the documented policy (retain via the marginal
    -- fallback, count as a source-data violation) now handles them instead.
    --
    -- The baseline wave omitted Checklist 3 (350 of the 810 DSE items): it is
    -- recorded as zero for every child on all three response types, no baseline
    -- total exceeds Checklists 1+2 = 460, and follow-up records carry non-zero
    -- Checklist 3 counts for children whose baseline total already exceeded 390.
    -- Its true administered ceiling is therefore 460, not 810 (see
    -- INCOMPLETE_ADMINISTRATION_CEILINGS).
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
    -- us_01 (Edgin): the English Down syndrome subset of the Edgin cohort.
    --
    -- Read from ``vocab_us_01`` (derived by ``scripts/build_us01_source.py`` from
    -- the item-level contributor files) rather than from the ``wordbank_child``
    -- by-child export, for two reasons the export cannot address:
    --
    --   * The export is age-truncated. Wordbank's download page calls
    --     ``get_administration_data()`` without ``filter_age = FALSE``, so every
    --     administration outside its instrument's registered age window is dropped
    --     before the page's age slider is even built. That cut the Edgin Down
    --     syndrome subset from 345 administrations to 194.
    --   * Four source administrations have every word item blank, which Wordbank
    --     scores as zero. Two are Down syndrome rows inside the window, and at 12
    --     months the export holds two ``(0, 0)`` rows of which only one is the empty
    --     form — so they are separable only at item level. They are excluded when
    --     the source CSV is built.
    --
    -- Administrations outside the form's registered age window are carried in the
    -- source. Those *above* the window are admitted -- for a Down syndrome cohort an
    -- early-vocabulary form given to an older child is developmentally appropriate,
    -- and they are this study's only comprehension observations between 19 and 27
    -- months. Those below its floor are dropped (FORM_AGE_FLOORS), as are children
    -- recorded only at the form ceiling (CEILING_ONLY_CHILD_STUDIES). Both rules read
    -- ``survey_vocab_max``, so the view's column list is untouched. See
    -- notes/202608031500-edgin-out-of-window-administrations.md.
    --
    -- Wordbank's CDI: Words & Sentences (WS) form records comprehension as a
    -- production proxy (comprehension == production by data convention), so
    -- understood is taken only from the genuinely bivariate forms — the same
    -- guard load_data applies on the TD side. WS rows still contribute
    -- production (spoken/produced). See
    -- notes/202607061200-us01-edgin-ws-comprehension-issue.md.
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
    -- us_03 (Fidler): Project CAPEabilities / Project EXPO, 396-word English
    -- Words and Gestures. `understood` is the inclusive comprehension total
    -- (the source's two mutually exclusive cells, already summed upstream), and
    -- `age` is the whole-month rounding of `age_months`, matching every other
    -- source. No sex was shared.
    --
    -- **The expressive cell is a produced union, not spoken.** The study
    -- authors state: "Understands and Says is inclusive of expressive language through spoken word and sign."
    -- The source document asserted a speech-only reading twice, both inferred
    -- from the column's original name, `spoken`; the column has since been
    -- renamed to `produced` here and upstream, and
    -- data/vocab_data_us_03.md carries the correction. Modalities cannot
    -- be separated -- there
    -- is one number, not the exclusive cells nz_01 and uk_07 carry -- so no
    -- spoken marginal can be recovered, and claiming one would put a produced
    -- union into `q = S/U`, the headline estimand of VG10, VG16, VG19, VG20 and
    -- VG22, for 254 of about 1,400 Down syndrome spoken observations.
    --
    -- So `spoken` is NULL and the count lands in `produced`. `signed` is NULL
    -- too: signing is inside the union rather than absent, and a zero would
    -- assert something the source does not say. us_03 therefore informs
    -- comprehension, and its production waits for a produced-outcome model --
    -- the rows are ordinary understood-without-spoken observations, which every
    -- engine already handles.
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
        810                                 as survey_vocab_max
    FROM vocab_ie_02 as vie2
    WHERE vie2.english_speaking = 'yes'
    UNION ALL
    -- nz_01 (Foster-Cohen): production-only, no comprehension. The CSV columns are
    -- modality-exclusive, so any-modality spoken = spoken + spoken_signed (a + c)
    -- and signed = signed + spoken_signed (b + c). 675-item NZCDI ceiling.
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
    -- es_01 (Galeote): a Spanish cross-sectional Down syndrome sample assessed on
    -- the 651-word CDI-Down, the Spanish MB-CDI adaptation for children with Down
    -- syndrome.
    --
    -- The source CSV also carries the study's 186 mental-age and sex matched
    -- typically developing children (group = 'TD'), which this Down syndrome
    -- relation excludes. They are a Spanish-normed comparison sample on a
    -- different instrument, so they are not interchangeable with the Wordbank TD
    -- reference pool load_data draws on, and pooling them would put a second
    -- instrument into the pool the Down syndrome exclusions are benchmarked
    -- against. They stay available in vocab_es_01 for a matched-pair analysis
    -- (pair_id links a DS child to its TD partner).
    --
    -- The CDI-Down adds a third response column for *symbolic* (referential)
    -- gestures -- "gestures representing specific lexical items" (Galeote et al.,
    -- 2011) -- so the source's `gestured` count is a gestural lexicon scored
    -- against the same 651 words, not a tally of generic communicative gestures.
    -- It is therefore read as this repository's `signed` construct: a non-vocal
    -- expressive lexicon recorded per word. Like uk_02 and nz_01 -- and unlike
    -- uk_01, see SIGNED_ONLY_STUDIES -- it is a TOTAL, counting words gestured
    -- whether or not they are also spoken, so it is comparable without item-level
    -- re-derivation. All 186 rows carry a non-zero total.
    --
    -- `produced` is the source's own recorded spoken-or-gestured union, each word
    -- counted once, so it is a de-duplicated union like uk_01's and nz_01's rather
    -- than a sum. It exceeds `spoken` by a mean of 28 words.
    --
    -- Guard: a gestural total larger than the union it belongs to is impossible --
    -- a union cannot be smaller than either of its parts -- so such a row's
    -- `signed` is masked rather than passed to the signing models as a total. One
    -- of the 186 rows is affected (1 word spoken, 15 gestured, union 11); which of
    -- its three source numbers is wrong cannot be determined, and its understood,
    -- spoken and produced values are unaffected. See data/vocab_data_es_01.md.
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
    -- uk_07 (PACT-DS; Burgoyne, Baxter, Hartwell, Pagnamenta & Stojanovik): a UK
    -- feasibility randomised controlled trial of a parent-delivered early language
    -- intervention. 30 children with Down syndrome, three assessment points each
    -- (83 retained rows, 34-95 months), on the 674-item "Reading CDI" -- the
    -- University of Reading adaptation, which adds a per-item sign coding.
    --
    -- Its expressive columns are modality-EXCLUSIVE cells, the nz_01 convention
    -- rather than the uk_01/ie_02/uk_04/uk_05 one: `spoken` is says-only (no
    -- sign), `signed` is signs-only (no says), `spoken_signed` is both. So the
    -- any-modality marginals are spoken = a + c and signed = b + c, and the
    -- source's `produced` is already the union of all three, each word once.
    -- `signed` is therefore a TOTAL sign count -- comparable with uk_02, nz_01 and
    -- es_01 without item-level re-derivation, and so not a SIGNED_ONLY_STUDIES
    -- case. 81 of the 83 rows carry a non-zero total, from all 30 children.
    --
    -- Unlike nz_01 this source also records comprehension, so every row carries
    -- `understood`. That makes uk_07 the second source after uk_02 supplying the
    -- four-cell WITHIN-UNDERSTOOD cross-tab that identifies the sign-speech
    -- association psi: understood_only = understood - produced, plus the three
    -- cells above. VG15 consumes those cells from the raw CSV (see
    -- common_joint_modality) and drops uk_07's marginals there to avoid double
    -- counting; this view's marginals feed every other model.
    --
    -- Both trial arms are pooled. The models describe vocabulary against age, not
    -- treatment effect, and the arm is a property of the child rather than of the
    -- measurement; `group` stays in vocab_uk_07 for a stratified analysis, and the
    -- intervention arm's T1-T3 growth being partly programme-driven is recorded as
    -- a source caveat.
    --
    -- One administration (58 months, 191 understood against 489 produced) is
    -- withheld pending clarification with the source team, and so is absent from
    -- vocab_uk_07 itself -- see UK07_WITHHELD_ADMINISTRATIONS. It is the only row
    -- in the source where production exceeds comprehension, and the only one whose
    -- understood_only cross-tab cell would be negative. 82 rows remain.
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
    """Return ``df`` in a canonical row order independent of scan order.

    The loader queries carry no ``ORDER BY``, so row order otherwise follows
    the DuckDB scan, which is not contractual and can change across versions
    and platforms. Everything statistical is order-invariant, but the fit
    manifest records an exact hash of the prepared frame — schema, values and
    row order — precisely so a stale posterior can be told from a current one,
    and a hash over a nondeterministic order cannot be recomputed for
    validation (issue #266 finding 1). Sorting on every column, stably and
    with NaNs last, gives a total order up to exact duplicate rows, whose
    relative order cannot change the hash.
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
    include_produced=False,
):
    """
    Load the combined data from the DuckDB database.

    (Run ./scripts/prepare_data.py to create the database if it doesn't exist.)

    Counts from partial administrations are masked by default
    (:func:`mask_incomplete_administrations`), because they are not on the
    810-item reference scale the model likelihoods assume. This is applied here
    rather than per-engine — unlike the signing-source masking, which only the
    signing models need — so every consumer of the DS pool gets the same scale.

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
            Read :data:`FORM_AGE_FLOORS` first — three of the 16 rows this puts back
            report 236 to 368 words *spoken* at 6 months.
        include_ceiling_only_children (bool): Reintroduce children whose every
            administration sits at their form's ceiling, for sensitivity analysis.
            Defaults to False. Read :data:`CEILING_ONLY_CHILD_STUDIES` first — this puts
            back 64 children and 98 administrations that carry a documented
            preparation-batch signature.
        include_comprehension_below_production (bool): Reintroduce comprehension
            counts that fall below the child's recorded ``produced`` union, for
            sensitivity analysis. Defaults to False.
        include_same_day_disagreements (bool): Reintroduce production counts
            contradicted by a same-day administration on another form, for
            sensitivity analysis. Defaults to False. Read
            :data:`SAME_DAY_DISAGREEMENT_FACTOR` first — the two counts this
            puts back record 385 and 406 words spoken at 23 months against
            same-day measurements of 11 and 50.
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
                -- Selected for mask_comprehension_below_production, and by
                -- default dropped before returning: `produced` is the union of
                -- the two modalities, which no registered model consumes, and
                -- adding a column to the returned frame would change what
                -- every caller sees. `include_produced` retains it on request.
                produced,
                survey_vocab_max
            FROM vocab_combined
            WHERE age <= $1
            """,
            [age_limit],
        ).df()
    df = _deterministic_row_order(df)

    # Both source-admissibility rules run before de-duplication and before any rule
    # that compares a child's records against each other. A ceiling-saturated row would
    # otherwise become the later value the longitudinal-collapse signature is measured
    # against, and could mask a valid count as a "collapse" that is an artefact of the
    # unusable row being present. exclude_ceiling_only_children must also see raw
    # counts, before the near-ceiling rule masks any of them, or its child-level test
    # would depend on the order the rules run in.
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
    df, _ = mask_duplicated_outcome_administrations(
        df, include_duplicated=include_duplicated_outcomes
    )
    df, _ = mask_implausible_production_administrations(
        df, include_implausible=include_implausible_production
    )
    # Same-day contradictions run after the production rules deliberately:
    # several ceiling-region same-day pairs are already masked by the
    # near-ceiling, collapse and duplicated-outcome signatures, so running
    # last-of-the-production-rules keeps those rules' documented counts
    # unchanged and makes this rule's catch exactly the counts nothing else
    # explains. The net masking is order-independent.
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
    # Last, and deliberately so: this rule compares two columns of a single row,
    # so it needs no cross-row context, and running it after the others means it
    # only fires on comprehension counts that survived every earlier rule. A row
    # whose `understood` an earlier mask already cleared is not counted twice.
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


def count_reinstated_implausible_production(max_age_months: int | None = None) -> int:
    """Spoken observations the implausible-production rule masks by default.

    This is what the ``include_implausible_production`` sensitivity puts back, and
    it exists so the sensitivity's own fit log can state the size of what it
    reinstated. A registered check that cannot be seen to have done anything is
    the failure the retired ``us01-ceiling-excluded`` variants exhibited; a
    reinstatement variant printing 0 here would be the same fault in mirror image.

    Derived by differencing the two loader paths rather than reimplementing the
    signature, so it cannot drift from the rule it reports on. The figure is
    the sensitivity's *net* reinstatement, not the implausible rule's own
    catch: :func:`mask_same_day_production_disagreements` independently
    re-masks reinstated ceiling-region counts that have an observed same-day
    partner, so the count is smaller than the rule's documented total (5
    against 11 in the current pool). Pass ``include_same_day_disagreements``
    as well to reinstate those too.
    """
    masked = load_combined_data(max_age_months=max_age_months)
    reinstated = load_combined_data(
        max_age_months=max_age_months, include_implausible_production=True
    )
    return int(
        reinstated["spoken"].notna().sum() - masked["spoken"].notna().sum()
    )


def _subsample_subjects(
    df: pd.DataFrame, sample_fraction: float, random_seed: int
) -> pd.DataFrame:
    """Draw a fraction of *subjects*, keeping all their administrations.

    Subsampling rows independently destroys the within-child replication that
    identifies a subject random effect. In the typically-developing pool a child
    contributes 1.32 administrations on average and 15.9% contribute more than
    one; drawing 10% of *rows* leaves 1.04 and 3.8%, at which point a subject
    random intercept and the observation-level Beta-Binomial dispersion are the
    same quantity — two per-observation noise terms with nothing to separate
    them.

    That is not hypothetical. Fitting VG11 to a 10% row-wise draw produces a
    posterior with two entirely separated modes — the between-child spread
    attributed either to ``kappa`` (``tau_subject`` about 0.07, ``a_kappa``
    about 1.2) or to the subject effects (``tau_subject`` about 1.08,
    ``a_kappa`` about 4.1) — with six chains splitting 3/3, no within-chain
    migration, and R-hat 1.72. The same model on a subject-wise draw of the same
    size is unimodal at R-hat 1.01. See
    ``notes/202608020829-kappa-and-eta-q-prior-recalibration.md`` §§11-12.

    Subjects are keyed by ``study``/``subject_id`` together, matching the
    ``subject_key`` convention the random-effect engines use, so a subject
    identifier repeated across datasets is not merged.

    The key list is **sorted** before sampling. ``Series.unique`` preserves order
    of first appearance and ``Series.sample`` draws by position, so an unsorted
    list would make the selected subjects depend on the order DuckDB happened to
    return rows in — the loader's query carries no ``ORDER BY``. Sorting matches
    what the random-effect engines already do when they assign study and subject
    codes, and makes the draw reproducible from the seed alone.
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
) -> pd.DataFrame:
    """
    Load vocabulary data for the specified population.

    Parameters
    ----------
    population : Population
        Which population to load data for.
    columns : list[str]
        Columns to select (e.g. ["age", "spoken"] or ["age", "understood", "spoken"]).
        For DS the ``study`` and ``subject_id`` columns are available.
        For TD the ``study`` column is aliased from ``dataset_name`` (the Wordbank
        dataset/lab identifier), along with ``subject_id``, ``form`` and ``language``.
    sample_fraction : float
        Fraction of **subjects** to subsample (TD only). 1.0 = no subsampling.
        Whole children are drawn and all their administrations kept, so
        within-child replication survives the subsample — see
        :func:`_subsample_subjects` for why drawing rows instead is unsafe for
        any model carrying subject random effects.
    random_seed : int
        Random seed for subsampling.
    max_age_months : int | None
        Upper bound on age (inclusive, months). None means no upper bound.
    languages : tuple[str, ...] | None
        Wordbank ``language`` values to include (TD only). Defaults to
        :data:`ENGLISH_LANGUAGES`. Pass a wider tuple to broaden the scope, or
        ``None`` to include all languages. Ignored for DS (the DS subset is
        fixed to English when the database is built).
    include_incomplete_administrations, include_duplicated_outcomes, include_implausible_production, include_below_form_floor, include_ceiling_only_children, include_comprehension_below_production, include_same_day_disagreements : bool
        Reinstate records that :func:`load_combined_data` masks or drops by default,
        for sensitivity analysis. **DS only** — each names a specific documented
        defect class in the DS pool, so passing one for the TD population is a
        caller error rather than a silent no-op.

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
    }
    if population == Population.DOWN_SYNDROME:
        df = load_combined_data(max_age_months=max_age_months, **reinstatements)
        return df[columns]

    if any(reinstatements.values()):
        raise ValueError(
            "Defect-reinstatement flags apply to the Down syndrome pool only; "
            f"got {sorted(k for k, v in reinstatements.items() if v)} for {population}."
        )

    # Typically developing — query wordbank_child directly.
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
                health_conditions
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
