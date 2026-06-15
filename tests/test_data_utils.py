# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

import duckdb
import pandas as pd

import vocab_growth.data_utils as data_utils
from vocab_growth.models.definitions import Population


def test_td_load_data_keeps_ws_as_spoken_only(tmp_path, monkeypatch):
    db_path = _create_wordbank_db(tmp_path)
    monkeypatch.setattr(data_utils, "VOCABULARY_DATA_PATH", str(db_path))

    df = data_utils.load_data(
        Population.TYPICALLY_DEVELOPING,
        columns=["form", "age", "understood", "spoken"],
    )

    df = df.sort_values("form").reset_index(drop=True)
    assert df["form"].to_list() == ["Oxford CDI", "WG", "WS"]

    ws_row = df.loc[df["form"] == "WS"].iloc[0]
    assert pd.isna(ws_row["understood"])
    assert ws_row["spoken"] == 51


def test_td_understood_data_excludes_ws_before_sampling(tmp_path, monkeypatch):
    db_path = _create_wordbank_db(tmp_path)
    monkeypatch.setattr(data_utils, "VOCABULARY_DATA_PATH", str(db_path))

    df = data_utils.load_data(
        Population.TYPICALLY_DEVELOPING,
        columns=["form", "age", "understood"],
    )

    assert sorted(df["form"].to_list()) == ["Oxford CDI", "WG"]


def test_td_spoken_data_includes_ws_and_bivariate_forms(tmp_path, monkeypatch):
    db_path = _create_wordbank_db(tmp_path)
    monkeypatch.setattr(data_utils, "VOCABULARY_DATA_PATH", str(db_path))

    df = data_utils.load_data(
        Population.TYPICALLY_DEVELOPING,
        columns=["form", "age", "spoken"],
    )

    assert sorted(df["form"].to_list()) == ["Oxford CDI", "WG", "WS"]


def test_td_load_data_defaults_to_english_only(tmp_path, monkeypatch):
    db_path = _create_wordbank_db(tmp_path)
    monkeypatch.setattr(data_utils, "VOCABULARY_DATA_PATH", str(db_path))

    df = data_utils.load_data(
        Population.TYPICALLY_DEVELOPING,
        columns=["language", "form", "age", "spoken"],
    )

    assert df["language"].unique().tolist() == ["English (British)"]


def test_td_load_data_can_widen_languages(tmp_path, monkeypatch):
    db_path = _create_wordbank_db(tmp_path)
    monkeypatch.setattr(data_utils, "VOCABULARY_DATA_PATH", str(db_path))

    df = data_utils.load_data(
        Population.TYPICALLY_DEVELOPING,
        columns=["language", "form", "age", "spoken"],
        languages=None,
    )

    assert "Norwegian" in df["language"].tolist()


def _create_wordbank_db(tmp_path):
    db_path = tmp_path / "vocabulary.duckdb"
    with duckdb.connect(str(db_path)) as con:
        con.execute(
            """
            CREATE TABLE wordbank_child (
                form VARCHAR,
                language VARCHAR,
                dataset_name VARCHAR,
                child_id VARCHAR,
                age DOUBLE,
                comprehension INTEGER,
                production INTEGER,
                typically_developing BOOLEAN,
                health_conditions VARCHAR
            )
            """
        )
        con.executemany(
            """
            INSERT INTO wordbank_child VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                ("WG",         "English (British)",  "Fenson (2007)",   "c01", 12.0,  40,  12, True,  None),
                ("Oxford CDI", "English (British)",  "Hamilton (2000)", "c02", 24.0, 190,  80, True,  None),
                ("WS",         "English (British)",  "Fenson (2007)",   "c03", 28.0,  51,  51, True,  None),
                ("WSShort",    "English (British)",  "Fenson (2007)",   "c04", 22.0,  20,  20, True,  None),
                ("TEDS Twos",  "English (British)",  "Fenson (2007)",   "c05", 24.0,  75,  75, True,  None),
                ("WG",         "English (British)",  "Fenson (2007)",   "c06", 35.0, 100,  50, True,  None),
                ("WG",         "English (British)",  "Fenson (2007)",   "c07", 18.0,  60,  20, True,  "premature"),
                ("WG",         "English (British)",  "Fenson (2007)",   "c08", 18.0,  60,  20, False, None),
                # Non-English row that otherwise matches: excluded by the default English filter.
                ("WG",         "Norwegian",           "Simonsen (2014)", "c09", 16.0,  55,  18, True,  None),
            ],
        )
    return db_path
