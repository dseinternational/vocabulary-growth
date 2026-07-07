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

US_01_MAX_PRODUCTION = 100
"""Production cap on the us_01/Edgin DS Wordbank subset.

Inherited from the initial import with no recorded rationale; rows above the
threshold are kept out until their source form and eligibility can be
revalidated (in the 2026-07 export it drops the highest-production
administrations: 8/87 WG and 24/109 WS English DS rows). See
``notes/202607061200-us01-edgin-ws-comprehension-issue.md``.
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
               WHEN vuk2.form = 'DSE' THEN 800
               WHEN vuk2.form = 'Oxford_CDI' THEN 428
               ELSE NULL
           END                as survey_vocab_max
    FROM vocab_uk_02 as vuk2
    UNION ALL
    SELECT 'ie_01'                                                   as study,
           vie.subject_id,
           NULL                                                        as sex,
           vie.age_months_start                                        as age,
           GREATEST(vie.says_total_start, vie.understands_total_start) as understood,
           vie.says_total_start                                        as spoken,
           null                                                        as signed,
           null                                                        as produced,
           800                                                         as survey_vocab_max
    FROM vocab_ie_01 as vie
    UNION ALL
    SELECT 'ie_01'                                               as study,
           vie.subject_id,
           NULL                                                    as sex,
           vie.age_months_end                                      as age,
           GREATEST(vie.says_total_end, vie.understands_total_end) as understood,
           vie.says_total_end                                      as spoken,
           null                                                    as signed,
           vie.says_total_end                                      as produced,
           800                                                     as survey_vocab_max
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
               WHEN 'WS' THEN 690
               ELSE NULL
               END                            as survey_vocab_max
    FROM wordbank_child
    WHERE dataset_name = 'Edgin'
      AND language IN ({english_sql_list})
      AND lower(health_conditions) = 'down syndrome'
      -- Legacy production cap (see US_01_MAX_PRODUCTION); kept as-is pending
      -- review of its rationale, per the 2026-07-06 note above.
      AND production <= {US_01_MAX_PRODUCTION}
    UNION ALL
    SELECT 'uk_03'                           as study,
           vuk2025.subject_id,
           NULL                                as sex,
           vuk2025.age,
           vuk2025.comprehension               as understood,
           vuk2025.production                  as spoken,
           null                                as signed,
           vuk2025.production                  as produced,
           418                                 as survey_vocab_max
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
        418                                 as survey_vocab_max
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
        418                                 as survey_vocab_max
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
        418                                 as survey_vocab_max
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
        800                                 as survey_vocab_max
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
        800                                 as survey_vocab_max
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


def load_combined_data(max_age_months=None):
    """
    Load the combined data from the DuckDB database.

    (Run ./scripts/prepare_data.py to create the database if it doesn't exist.)

    Parameters:
    -----------
        max_age_months (int, optional): The maximum age in months to include in the data. Defaults to None (no limit).

    Returns:
    --------
        pd.DataFrame: The combined data as a DataFrame.
    """
    age_limit = max_age_months if max_age_months is not None else 1200

    with duckdb.connect(VOCABULARY_DATA_PATH) as con:
        df = con.execute(
            """
            SELECT
                study,
                subject_id,
                sex,
                age,
                understood,
                spoken,
                signed
            FROM vocab_combined
            WHERE age <= $1
            """,
            [age_limit],
        ).df()

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

    with duckdb.connect(VOCABULARY_DATA_PATH) as con:
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
