# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

# run from repository root with `python scripts/prepare_data.py`

import os
import time

import duckdb
import pandas as pd

from vocab_growth.data_utils import (
    drop_ie02_withheld_administrations,
    drop_uk01_withheld_subjects,
    drop_uk07_withheld_administrations,
    vocab_combined_view_sql,
)
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
    "vocab_es_01": "./data/vocab_data_es_01.csv",
    "vocab_uk_07": "./data/vocab_data_uk_07.csv",
    "vocab_us_01": "./data/vocab_data_us_01.csv",
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

# Withhold the ie_02 t2 administration whose counts are internally
# contradictory — a 331-word comprehension surge, a 237-word signing surge and
# a 96% speech collapse asserted for the same three months, the pattern of a
# checklist completed differently between waves. Dropped here at load so it is
# absent from the merged CSV, the DuckDB vocab_ie_02 table and the
# vocab_combined view alike; the child's t1 administration is retained. See
# data_utils.IE02_WITHHELD_ADMINISTRATIONS for the evidence and how to
# reinstate.
vocab_ie_02_df, _ie02_withheld = drop_ie02_withheld_administrations(vocab_ie_02_df)
console.print(
    f"[yellow]ie_02 administrations withheld as internally contradictory:[/yellow] "
    f"{_ie02_withheld}"
)
vocab_it_01_df = _loaded["vocab_it_01"]

# Exclude the uk_01 subjects withheld as probable homonym fusions. The source
# keys children by name alone, and ID_E33ADE657109EBB8's four rows interleave
# two contradictory modality profiles (a signer who barely speaks, a speaker
# who never signs) — the signature of two same-named children fused under one
# id, and the origin of the −424-word "collapse" at 76–78 months. Dropped here
# at load so the rows are absent from the merged CSV, the DuckDB vocab_uk_01
# table and the vocab_combined view alike. See
# data_utils.UK01_WITHHELD_SUBJECTS for the evidence and how to reinstate.
vocab_uk_01_df, _uk01_withheld = drop_uk01_withheld_subjects(_loaded["vocab_uk_01"])
console.print(
    f"[yellow]uk_01 rows withheld as probable homonym fusions:[/yellow] "
    f"{_uk01_withheld}"
)
vocab_uk_02_df = _loaded["vocab_uk_02"]
vocab_uk_03_df = _loaded["vocab_uk_03"]
vocab_uk_04_df = _loaded["vocab_uk_04"]
vocab_uk_05_df = _loaded["vocab_uk_05"]
vocab_us_02_df = _loaded["vocab_us_02"]
vocab_uk_06_df = _loaded["vocab_uk_06"]
vocab_es_01_df = _loaded["vocab_es_01"]

# Exclude the uk_07 administrations withheld pending clarification with the source
# team — one row at 58 months recording 191 words understood against 489 produced,
# the only row in the source where production exceeds comprehension, at the end of
# a reported comprehension decline. Dropped here at load so it is absent from the
# merged CSV, the DuckDB vocab_uk_07 table and the vocab_combined view alike; the
# same helper guards VG15's cross-tab path, which reads this CSV directly. See
# data_utils.UK07_WITHHELD_ADMINISTRATIONS for the reasoning and how to reinstate.
vocab_uk_07_df, _uk07_withheld = drop_uk07_withheld_administrations(
    _loaded["vocab_uk_07"]
)
console.print(
    f"[yellow]uk_07 administrations withheld pending source clarification:[/yellow] "
    f"{_uk07_withheld}"
)

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

# Spain (es_01, Galeote): a cross-sectional sample of 186 children with Down
# syndrome and 186 mental-age/sex matched typically developing children, assessed
# on the 651-word CDI-Down. Only the Down syndrome children enter this relation —
# the TD children are a Spanish-normed comparison sample on a different instrument
# from the Wordbank TD pool (see vocab_combined_view_sql). This pre-guard concat
# carries understood/spoken only; the CDI-Down's symbolic-gesture lexicon reaches
# the models as `signed`, and its spoken-or-gestured union as `produced`, through
# the vocab_combined view.
es_01_to_merge = vocab_es_01_df.loc[
    vocab_es_01_df["group"] == "DS", ["subject_id", "age", "understood", "spoken"]
].copy()
es_01_to_merge["study"] = 12

# UK 7 (uk_07, PACT-DS): a longitudinal Down syndrome dataset from a feasibility
# RCT, three time points per child on a 674-item CDI. Its expressive columns are
# modality-exclusive cells (says-only / signs-only / both), following the nz_01
# convention, so the any-modality spoken marginal is word-only + both
# (spoken + spoken_signed). Both trial arms are pooled here; `group` stays in the
# vocab_uk_07 table for a stratified analysis. The signed marginal and the
# produced union reach the models through the vocab_combined view.
uk_07_to_merge = vocab_uk_07_df[["subject_id", "age", "understood"]].copy()
uk_07_to_merge["spoken"] = vocab_uk_07_df["spoken"] + vocab_uk_07_df["spoken_signed"]
uk_07_to_merge["study"] = 13


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
        es_01_to_merge,
        uk_07_to_merge,
    ],
    ignore_index=True,
)

console.print(
    f"[green]Raw merged rows (pre-guard):[/green] {len(merged_df):,} rows × {len(merged_df.columns)} cols"
)

# NB: vocab_data_merged.csv is written further down FROM the vocab_combined view
# (post form-ceiling guard and us_01 WS-comprehension guard), not this raw
# pre-guard concat (issue #131). The view is still not what the models read:
# load_combined_data applies further exclusion/masking rules on top of it — see
# the note at the export below.

console.print("[green]Creating DuckDB database and tables…[/green]")

db_path = "./data/vocabulary.duckdb"

# Build into a temporary file beside the target and swap it in only once the
# build has completed, so a failure partway through (a malformed source CSV, the
# large Wordbank read) cannot leave the existing database destroyed. Clear any
# stale temporary file — and DuckDB's write-ahead log beside it — from a
# previous failed run first.
db_tmp = db_path + ".tmp"
for _stale in (db_tmp, db_tmp + ".wal"):
    if os.path.exists(_stale):
        os.remove(_stale)

con = duckdb.connect(db_tmp)

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


con.execute(
    """
    CREATE TABLE vocab_es_01 AS
    SELECT * FROM vocab_es_01_df
    """
)

con.execute(
    """
    CREATE TABLE vocab_uk_07 AS
    SELECT * FROM vocab_uk_07_df
    """
)

# us_01 (Edgin) is derived from the item-level contributor files by
# scripts/build_us01_source.py, not read out of the by-child Wordbank export: that
# export is age-truncated by Wordbank's own download page and cannot separate the
# four all-blank administrations it scores as zeros. The table carries all four
# developmental-status groups; the vocab_combined view selects Down syndrome, and
# the comparison group stays available for a matched analysis.
vocab_us_01_df = _loaded["vocab_us_01"]

con.execute(
    """
    CREATE TABLE vocab_us_01 AS
    SELECT * FROM vocab_us_01_df
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

# Export the guarded vocab_combined view so vocab_data_merged.csv reflects the
# view rather than the raw pre-guard concat (issue #131 §3). This is NOT what
# the models consume: load_combined_data applies a further seven
# exclusion/masking rules in Python on top of the view (ceiling-only children,
# below-form-floor, duplicate administrations, and the four masking rules) and
# drops the `produced` column, so the CSV is a superset of the modelled data
# with outcome values the loader masks.
analysis_df = con.execute("SELECT * FROM vocab_combined").df()

# Same write-to-temporary-then-replace pattern as the database, so a failed
# export cannot truncate the existing CSV.
csv_path = "./data/vocab_data_merged.csv"
csv_tmp = csv_path + ".tmp"
analysis_df.to_csv(csv_tmp, index=False)
os.replace(csv_tmp, csv_path)

con.close()

os.replace(db_tmp, db_path)

key_value_table(
    "Data preparation complete",
    [
        ("Merged CSV (analysis view)", "./data/vocab_data_merged.csv"),
        ("DuckDB database", db_path),
        ("Raw merged rows (pre-guard)", f"{len(merged_df):,}"),
        ("Analysis rows (post-guard view)", f"{len(analysis_df):,}"),
        ("Elapsed", format_duration(time.perf_counter() - _started)),
    ],
)
