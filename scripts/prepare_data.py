# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

# run from repository root with `python scripts/prepare_data.py`

import os
import time

import duckdb
import pandas as pd

from vocab_growth.data_utils import ENGLISH_LANGUAGES
from vocab_growth.reporting import (
    console,
    format_duration,
    heading,
    key_value_table,
)

# SQL list literal of English Wordbank ``language`` values. The Wordbank export
# now contains all languages; the DS (Edgin) subset is restricted to English.
_ENGLISH_SQL_LIST = ", ".join(f"'{lang}'" for lang in ENGLISH_LANGUAGES)

# Data-quality guard for the Edgin DS Wordbank subset. Rows above this threshold
# are kept out until their source form and eligibility can be revalidated.
US_01_MAX_PRODUCTION = 100

_started = time.perf_counter()
heading("Preparing vocabulary data")

_sources = {
    "vocab_ie_01": "./data/vocab_data_ie_01.csv",
    "vocab_ie_02": "./data/vocab_data_ie_02.csv",
    "vocab_it_01": "./data/vocab_data_it_01.csv",
    "vocab_uk_01": "./data/vocab_data_uk_01.csv",
    "vocab_uk_02": "./data/vocab_data_uk_02.csv",
    "vocab_uk_03": "./data/vocab_data_uk_03.csv",
    "vocab_uk_04": "./data/vocab_data_uk_04.csv",
    "vocab_uk_05": "./data/vocab_data_uk_05.csv",
    "vocab_us_02": "./data/vocab_data_us_02.csv",
    "vocab_uk_06": "./data/vocab_data_uk_06.csv",
    "vocab_nz_01": "./data/vocab_data_nz_01.csv",
}
# nz_01 (Foster-Cohen) is added with the real anonymisation key in a separate
# data commit; tolerate its absence so the pipeline still builds without it.
_loaded = {
    name: pd.read_csv(path)
    for name, path in _sources.items()
    if name != "vocab_nz_01" or os.path.exists(path)
}
have_nz01 = "vocab_nz_01" in _loaded

key_value_table(
    "Loaded datasets",
    [(name, f"{len(df):,} rows × {len(df.columns)} cols") for name, df in _loaded.items()],
    value_header="Shape",
)

vocab_ie_01_df = _loaded["vocab_ie_01"]
vocab_ie_02_df = _loaded["vocab_ie_02"]

# Exclude ie_02 subject ID_79C464EF367C4D5B: an evident data-entry error. Both of
# its rows report near-ceiling counts implausible for the recorded ages (spoken/
# understood 432/477 at 13 mo and 456/477 at 16 mo, versus 0-38 spoken for every
# other ie_02 child under 24 mo), with the imitates/spoken/says_clearly columns
# byte-identical at both visits and understood pinned at 477 across the 3-month
# gap — the signature of one value propagated across the production columns. It is
# the sole source of the anomalous >400-words-before-20-months points in the
# spoken (VG01) and understood (VG02) trajectories. Dropped here at load so it is
# absent from both the merged CSV and the DuckDB vocab_ie_02 table (and hence the
# vocab_combined view the models read). Excluded pending source verification with
# the ie_02 data provider.
vocab_ie_02_df = vocab_ie_02_df[
    vocab_ie_02_df["subject_id"] != "ID_79C464EF367C4D5B"
].copy()
vocab_it_01_df = _loaded["vocab_it_01"]
vocab_uk_01_df = _loaded["vocab_uk_01"]
vocab_uk_02_df = _loaded["vocab_uk_02"]
vocab_uk_03_df = _loaded["vocab_uk_03"]
vocab_uk_04_df = _loaded["vocab_uk_04"]
vocab_uk_05_df = _loaded["vocab_uk_05"]
vocab_us_02_df = _loaded["vocab_us_02"]
vocab_uk_06_df = _loaded["vocab_uk_06"]

# Prepare the data for merging
vocab_to_merge = vocab_uk_01_df[["subject_id", "age", "understood", "spoken"]].copy()
vocab_to_merge["study"] = 1

edg_early_read_to_merge = (
    vocab_uk_02_df[["subject_id", "age", "comprehension", "spoken"]]
    .rename(columns={"comprehension": "understood"})
    .copy()
)
edg_early_read_to_merge["study"] = 2

ireland_t1_to_merge = (
    vocab_ie_01_df[
        [
            "subject_id",
            "age_months_start",
            "understands_total_start",
            "says_total_start",
        ]
    ]
    .rename(
        columns={
            "age_months_start": "age",
            "understands_total_start": "understood",
            "says_total_start": "spoken",
        }
    )
    .copy()
)
ireland_t1_to_merge["study"] = 3

ireland_t2_to_merge = (
    vocab_ie_01_df[
        ["subject_id", "age_months_end", "understands_total_end", "says_total_end"]
    ]
    .rename(
        columns={
            "age_months_end": "age",
            "understands_total_end": "understood",
            "says_total_end": "spoken",
        }
    )
    .copy()
)
ireland_t2_to_merge["study"] = 3

vocab_uk_03_to_merge = vocab_uk_03_df.copy()
vocab_uk_03_to_merge["study"] = 4

vocab_it_01_to_merge = vocab_it_01_df.copy()
vocab_it_01_to_merge["study"] = 5

vocab_uk_04_to_merge = vocab_uk_04_df.copy()
vocab_uk_04_to_merge["study"] = 6

vocab_uk_05_to_merge = vocab_uk_05_df.copy()
vocab_uk_05_to_merge["study"] = 7

vocab_us_02_to_merge = vocab_us_02_df.copy()
vocab_us_02_to_merge["study"] = 8

vocab_uk_06_to_merge = vocab_uk_06_df.copy()
vocab_uk_06_to_merge["study"] = 9

# Ireland 2 (ie_02): a longitudinal Down syndrome dataset already in long format
# (one row per timepoint t1/t2), carrying understood/spoken/signed counts. The
# instruments measure English vocabulary, so non-English-speaking children are
# excluded (english_speaking == 'yes'). Both recruitment groups are pooled as DS.
# (Subject ID_79C464EF367C4D5B was dropped at load — see the exclusion note where
# vocab_ie_02_df is read — so it is absent from both the merged CSV and the
# DuckDB vocab_ie_02 table / vocab_combined view.)
ireland_2_to_merge = (
    vocab_ie_02_df.loc[
        vocab_ie_02_df["english_speaking"] == "yes",
        ["subject_id", "age", "understood", "spoken"],
    ].copy()
)
ireland_2_to_merge["study"] = 10

# New Zealand (nz_01, Foster-Cohen): a longitudinal Down syndrome dataset,
# production-only (no comprehension) with a modality partition. The any-modality
# spoken marginal is word-only + both (spoken + spoken_signed); understood is
# unavailable. (VG15 instead consumes nz_01's produced cross-tab directly — see
# common_joint_modality — so this marginal feeds the other DS models.) The
# real-key CSV lands in a separate data commit; until then nz_01 is skipped.
if have_nz01:
    vocab_nz_01_df = _loaded["vocab_nz_01"]
    nz_01_to_merge = vocab_nz_01_df[["subject_id", "age"]].copy()
    nz_01_to_merge["understood"] = pd.NA
    nz_01_to_merge["spoken"] = (
        vocab_nz_01_df["spoken"] + vocab_nz_01_df["spoken_signed"]
    )
    nz_01_to_merge["study"] = 11
else:
    nz_01_to_merge = pd.DataFrame(
        columns=["subject_id", "age", "understood", "spoken", "study"]
    )


merged_df = pd.concat(
    [
        vocab_to_merge,
        edg_early_read_to_merge,
        ireland_t1_to_merge,
        ireland_t2_to_merge,
        vocab_uk_03_to_merge,
        vocab_it_01_to_merge,
        vocab_uk_04_to_merge,
        vocab_uk_05_to_merge,
        vocab_us_02_to_merge,
        vocab_uk_06_to_merge,
        ireland_2_to_merge,
        nz_01_to_merge,
    ],
    ignore_index=True,
)

console.print(
    f"[green]Merged dataset:[/green] {len(merged_df):,} rows × {len(merged_df.columns)} cols"
)
console.print("[green]Saving merged dataset to CSV…[/green]")

merged_df.to_csv("./data/vocab_data_merged.csv", index=False)

console.print("[green]Creating DuckDB database and tables…[/green]")

db_path = "./data/vocabulary.duckdb"

if os.path.exists(db_path):
    os.remove(db_path)

con = duckdb.connect(db_path)

con.execute(
    """
    CREATE TABLE vocab_uk_01 AS
    SELECT * FROM vocab_uk_01_df
    """
)

con.execute(
    """
    CREATE TABLE vocab_uk_02 AS
    SELECT * FROM vocab_uk_02_df
    """
)

con.execute(
    """
    CREATE TABLE vocab_ie_01 AS
    SELECT * FROM vocab_ie_01_df
    """
)

con.execute(
    """
    CREATE TABLE vocab_uk_03 AS
    SELECT * FROM vocab_uk_03_df
    """
)

con.execute(
    """
    CREATE TABLE vocab_it_01 AS
    SELECT * FROM vocab_it_01_df
    """
)

con.execute(
    """
    CREATE TABLE vocab_uk_04 AS
    SELECT * FROM vocab_uk_04_df
    """
)

con.execute(
    """
    CREATE TABLE vocab_uk_05 AS
    SELECT * FROM vocab_uk_05_df
    """
)

con.execute(
    """
    CREATE TABLE vocab_us_02 AS
    SELECT * FROM vocab_us_02_df
    """
)

con.execute(
    """
    CREATE TABLE vocab_uk_06 AS
    SELECT * FROM vocab_uk_06_df
    """
)

con.execute(
    """
    CREATE TABLE vocab_ie_02 AS
    SELECT * FROM vocab_ie_02_df
    """
)

if have_nz01:
    con.execute(
        """
        CREATE TABLE vocab_nz_01 AS
        SELECT * FROM vocab_nz_01_df
        """
    )
else:
    # Empty table with the nz_01 schema so the vocab_combined UNION still
    # resolves (contributes zero rows until the real-key CSV is committed).
    con.execute(
        """
        CREATE TABLE vocab_nz_01 (
            subject_id VARCHAR,
            age BIGINT,
            not_spoken_or_signed BIGINT,
            signed BIGINT,
            spoken_signed BIGINT,
            spoken BIGINT
        )
        """
    )


wordbank_child_df = pd.read_csv(
    "./data/wordbank_administration_data.csv", low_memory=False
)

con.execute(
    """
    CREATE TABLE wordbank_child AS
    SELECT * FROM wordbank_child_df
    """
)

con.execute(
    f"""
    CREATE VIEW vocab_combined AS
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
    SELECT 'us_01'                          as study,
           concat('id_', hex(hash(child_id))) as subject_id,
           sex,
           age,
           comprehension                      as understood,
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
      AND language IN ({_ENGLISH_SQL_LIST})
      AND lower(health_conditions) = 'down syndrome'
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

    """
)

con.close()

key_value_table(
    "Data preparation complete",
    [
        ("Merged CSV", "./data/vocab_data_merged.csv"),
        ("DuckDB database", db_path),
        ("Total merged rows", f"{len(merged_df):,}"),
        ("Elapsed", format_duration(time.perf_counter() - _started)),
    ],
)
