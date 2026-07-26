# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import os

import duckdb
import pandas as pd

import vocab_growth.environment as local_env
from vocab_growth.models.definitions import Population

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

SIGNED_ONLY_STUDIES = ("uk_01",)
"""Studies whose ``signed`` field excludes words that are also spoken.

The signing models estimate total sign use, so these fields are not comparable
without item-level re-derivation.  Keep the source rows for understood/spoken
outcomes while masking only their ``signed`` value by default.
"""

UNCERTAIN_SIGN_STUDIES = ("uk_06",)
"""Studies whose signing-field construct has not yet been source-verified."""

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


ENGLISH_LANGUAGES = (
    "English (American)",
    "English (Australian)",
    "English (British)",
    "English (Irish)",
)
"""Wordbank ``language`` values treated as English — the current default scope.

The ``wordbank_child`` table now holds the full multi-language Wordbank export.
Queries restrict to these English variants by default; pass a wider ``languages``
set (or ``None`` for all languages) to the loaders to widen the scope later.
"""


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

**Near-ceiling saturation.** ``spoken >= IMPLAUSIBLE_PRODUCTION_CEILING_FRACTION *
survey_vocab_max``. In ``us_01`` this matches 23 administrations: 18 at exactly the
680-item Words & Sentences ceiling, three just below (641, 656, 668), and the Words
& Gestures record at 396. Thirteen of them form a **contiguous child-id block**
(81207-81241) in which every record sits at exactly 680, no child has any other
administration, and the next id present is 81322 — an 81-id gap. That is a
preparation-batch signature, not thirteen exceptionally able children. For scale,
no typically-developing child of 1,469 aged 16-19 months reaches the Words &
Sentences ceiling, and their maximum is 643.

**Longitudinal collapse.** A count of at least ``COLLAPSE_MIN_VALUE`` that exceeds
the same child's later count by a factor of ``COLLAPSE_FACTOR`` or more. Vocabulary
does not shrink, so this is unambiguous — 656 words at 17 months against 12 at 23
months is not measurement noise. The floor matters: without it the rule fires on
trivial pairs such as 5 understood words falling to 1. The age scope matters too:
at older ages a decline can arise from a form change or from noise in large counts,
and two such records outside ``us_01`` (a uk_01 record at 76 months, an ie_02
record at 45) are deliberately left for separate investigation rather than masked
by a rule whose justification is developmental.

Within the scoped window the two signatures together match 30 ``us_01``
administrations and nothing in any other study. Not caught, and retained: a Words &
Sentences record of 406 words at 23 months with no later administration to
contradict it. It is extreme against the external benchmark but has no positive
defect signature, so it is a sensitivity target rather than an exclusion — the same
treatment as the high-comprehension records described on
:data:`DUPLICATED_OUTCOME_RATIO`.
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
#   - NZCDI (nz_01) = 675. uk_01 and it_01 carry a per-row source ceiling.
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

    The Wordbank export contains all languages; the DS (Edgin) subset is
    restricted to English via :data:`ENGLISH_LANGUAGES`.
    """
    english_sql_list = _sql_string_list(ENGLISH_LANGUAGES)
    bivariate_forms_sql_list = _sql_string_list(WORDBANK_BIVARIATE_FORMS)
    return f"""
    CREATE VIEW vocab_combined AS
    SELECT * FROM (
    SELECT 'uk_01' as study,
           vuk1.subject_id,
           vuk1.sex,
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
           CASE
               WHEN vuk2.gender = 0 THEN 'M'
               WHEN vuk2.gender = 1 THEN 'F'
               ELSE NULL
           END            as sex,
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
    -- us_01 (Edgin): the English Down syndrome subset of the Wordbank export.
    -- Wordbank's CDI: Words & Sentences (WS) form records comprehension as a
    -- production proxy (comprehension == production by data convention), so
    -- understood is taken only from the genuinely bivariate forms — the same
    -- guard load_data applies on the TD side. WS rows still contribute
    -- production (spoken/produced). See
    -- notes/202607061200-us01-edgin-ws-comprehension-issue.md.
    SELECT 'us_01'                          as study,
           concat('id_', hex(hash(child_id))) as subject_id,
           sex,
           age,
           CASE
               WHEN form IN ({bivariate_forms_sql_list}) THEN comprehension
               ELSE NULL
           END                                as understood,
           production                         as spoken,
           null                               as signed,
           production                         as produced,
           CASE form
               WHEN 'WG' THEN 396
               WHEN 'WS' THEN 680
               ELSE NULL
               END                            as survey_vocab_max
    FROM wordbank_child
    WHERE dataset_name = 'Edgin'
      AND language IN ({english_sql_list})
      AND lower(health_conditions) = 'down syndrome'
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
        NULL                                as sex,
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
        NULL                                as sex,
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
        NULL                                as sex,
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
    ) vc
    WHERE {_CEILING_GUARD_KEEP}
    """


def load_combined_data(
    max_age_months=None,
    *,
    include_incomplete_administrations=False,
    include_duplicated_outcomes=False,
    include_implausible_production=False,
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
                survey_vocab_max
            FROM vocab_combined
            WHERE age <= $1
            """,
            [age_limit],
        ).df()

    # De-duplicate first, so a repeated row cannot affect the within-child
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
    return df


def load_data(
    population: Population,
    columns: list[str],
    sample_fraction: float = 1.0,
    random_seed: int = 47,
    max_age_months: int | None = None,
    languages: tuple[str, ...] | None = ENGLISH_LANGUAGES,
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
        Fraction of data to subsample (TD only). 1.0 = no subsampling.
    random_seed : int
        Random seed for subsampling.
    max_age_months : int | None
        Upper bound on age (inclusive, months). None means no upper bound.
    languages : tuple[str, ...] | None
        Wordbank ``language`` values to include (TD only). Defaults to
        :data:`ENGLISH_LANGUAGES`. Pass a wider tuple to broaden the scope, or
        ``None`` to include all languages. Ignored for DS (the DS subset is
        fixed to English when the database is built).

    Returns
    -------
    pd.DataFrame
        DataFrame with the requested columns.
    """
    if population == Population.DOWN_SYNDROME:
        df = load_combined_data(max_age_months=max_age_months)
        return df[columns]

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

    age_upper = max_age_months if max_age_months is not None else 30

    params: list = [td_forms, age_upper]
    language_clause = ""
    if languages is not None:
        params.append(list(languages))
        language_clause = f"AND language IN ${len(params)}"

    with duckdb.connect(VOCABULARY_DATA_PATH, read_only=True) as con:
        td_df = (
            con.execute(
                f"""
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
            FROM wordbank_child
            WHERE typically_developing = true
                AND age <= $2
                AND health_conditions IS NULL
                AND dataset_name NOT IN ({_sql_string_list(TD_POOL_EXCLUDED_DATASETS)})
                AND form IN $1
                {language_clause}
            """,
                params,
            )
            .df()
        )

    if sample_fraction < 1.0:
        td_df = (
            td_df.sample(frac=sample_fraction, random_state=random_seed)
            .reset_index(drop=True)
        )

    return td_df[columns]
