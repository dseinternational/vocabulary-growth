# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

# run from repository root with `python scripts/prepare_data.py`

import os
import time

import duckdb
import pandas as pd

from vocab_growth.data_utils import vocab_combined_view_sql
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

# The view definition lives in vocab_growth.data_utils so the per-study
# transformations (notably the us_01/Edgin Wordbank form guard) are importable
# and regression-tested alongside load_combined_data.
con.execute(vocab_combined_view_sql())

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
