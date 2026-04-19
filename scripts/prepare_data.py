# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

# run from repository root with `python scripts/prepare_data.py`

import os
import time

import duckdb
import pandas as pd

from vocab_growth.reporting import (
    console,
    format_duration,
    heading,
    key_value_table,
)

_started = time.perf_counter()
heading("Preparing vocabulary data")

_sources = {
    "vocab_ie_01": "./data/vocab_data_ie_01.csv",
    "vocab_it_01": "./data/vocab_data_it_01.csv",
    "vocab_uk_01": "./data/vocab_data_uk_01.csv",
    "vocab_uk_02": "./data/vocab_data_uk_02.csv",
    "vocab_uk_03": "./data/vocab_data_uk_03.csv",
    "vocab_uk_04": "./data/vocab_data_uk_04.csv",
    "vocab_uk_05": "./data/vocab_data_uk_05.csv",
    "vocab_us_02": "./data/vocab_data_us_02.csv",
    "vocab_uk_06": "./data/vocab_data_uk_06.csv",
}
_loaded = {name: pd.read_csv(path) for name, path in _sources.items()}

key_value_table(
    "Loaded datasets",
    [(name, f"{len(df):,} rows × {len(df.columns)} cols") for name, df in _loaded.items()],
    value_header="Shape",
)

vocab_ie_01_df = _loaded["vocab_ie_01"]
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


wordbank_child_df = pd.read_csv(
    "./data/wordbank_administration_data_en.csv", low_memory=False
)

con.execute(
    """
    CREATE TABLE wordbank_child AS
    SELECT * FROM wordbank_child_df
    """
)

con.execute(
    """
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
    UNION
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
    UNION
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
    UNION
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
    UNION
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
      AND lower(health_conditions) = 'down syndrome'
      AND production <= 100
    UNION
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
    UNION
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
    UNION
    SELECT 'uk_04'                           as study,
        vuk2013.subject_id,
        NULL                                as sex,
        vuk2013.age,
        vuk2013.understood,
        vuk2013.spoken,
        vuk2013.signed,
        vuk2013.spoken                      as produced,
        418                                 as survey_vocab_max -- TODO: confirm for Reading Communicative Development Inventories
    FROM vocab_uk_04 as vuk2013
        UNION
    SELECT 'uk_05'                           as study,
        vuk05.subject_id,
        NULL                                as sex,
        vuk05.age,
        vuk05.understood,
        vuk05.spoken,
        vuk05.signed,
        vuk05.spoken                      as produced,
        418                                 as survey_vocab_max -- TODO: confirm for Reading Communicative Development Inventories
    FROM vocab_uk_05 as vuk05
        UNION
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
        UNION
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
