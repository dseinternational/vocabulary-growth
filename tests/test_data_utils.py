# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

import duckdb
import pandas as pd

import vocab_growth.data_utils as data_utils
from vocab_growth.models.definitions import Population


def test_td_bivariate_data_excludes_ws_before_sampling(tmp_path, monkeypatch):
    db_path = _create_vocab_db(tmp_path)
    monkeypatch.setattr(data_utils, "VOCABULARY_DATA_PATH", str(db_path))

    df = data_utils.load_data(
        Population.TYPICALLY_DEVELOPING,
        columns=["form", "age", "understood", "spoken"],
    )

    df = df.sort_values("form").reset_index(drop=True)
    assert df["form"].to_list() == ["Oxford CDI", "WG"]


def test_td_understood_data_excludes_ws_before_sampling(tmp_path, monkeypatch):
    db_path = _create_vocab_db(tmp_path)
    monkeypatch.setattr(data_utils, "VOCABULARY_DATA_PATH", str(db_path))

    df = data_utils.load_data(
        Population.TYPICALLY_DEVELOPING,
        columns=["form", "age", "understood"],
    )

    assert sorted(df["form"].to_list()) == ["Oxford CDI", "WG"]


def test_td_spoken_data_includes_ws_and_bivariate_forms(tmp_path, monkeypatch):
    db_path = _create_vocab_db(tmp_path)
    monkeypatch.setattr(data_utils, "VOCABULARY_DATA_PATH", str(db_path))

    df = data_utils.load_data(
        Population.TYPICALLY_DEVELOPING,
        columns=["form", "age", "spoken"],
    )

    assert sorted(df["form"].to_list()) == ["Oxford CDI", "WG", "WS"]


def test_td_load_data_defaults_to_english_only(tmp_path, monkeypatch):
    db_path = _create_vocab_db(tmp_path)
    monkeypatch.setattr(data_utils, "VOCABULARY_DATA_PATH", str(db_path))

    df = data_utils.load_data(
        Population.TYPICALLY_DEVELOPING,
        columns=["language", "form", "age", "spoken"],
    )

    assert df["language"].unique().tolist() == ["English (British)"]


def test_td_load_data_can_widen_languages(tmp_path, monkeypatch):
    db_path = _create_vocab_db(tmp_path)
    monkeypatch.setattr(data_utils, "VOCABULARY_DATA_PATH", str(db_path))

    df = data_utils.load_data(
        Population.TYPICALLY_DEVELOPING,
        columns=["language", "form", "age", "spoken"],
        languages=None,
    )

    assert "Norwegian" in df["language"].tolist()


def _load_us01(tmp_path, monkeypatch):
    db_path = _create_vocab_db(tmp_path)
    monkeypatch.setattr(data_utils, "VOCABULARY_DATA_PATH", str(db_path))
    df = data_utils.load_combined_data()
    return df[df["study"] == "us_01"]


def test_ds_us01_ws_form_contributes_spoken_but_not_understood(tmp_path, monkeypatch):
    us01 = _load_us01(tmp_path, monkeypatch)

    ws_rows = us01[us01["spoken"] == 77]
    assert len(ws_rows) == 1
    assert ws_rows["understood"].isna().all()


def test_ds_us01_wg_form_keeps_independent_comprehension(tmp_path, monkeypatch):
    us01 = _load_us01(tmp_path, monkeypatch)

    wg_rows = us01[us01["spoken"] == 12]
    assert wg_rows["understood"].tolist() == [40]


def test_ds_us01_no_row_carries_the_ws_production_proxy_as_understood(
    tmp_path, monkeypatch
):
    us01 = _load_us01(tmp_path, monkeypatch)

    bivariate = us01.dropna(subset=["understood"])
    assert not bivariate.empty  # the WG row is retained
    assert not (bivariate["understood"] == bivariate["spoken"]).any()


def test_ds_us01_production_cap_excludes_rows_above_100(tmp_path, monkeypatch):
    us01 = _load_us01(tmp_path, monkeypatch)

    assert sorted(us01["spoken"].tolist()) == [12, 77]


def _study_frame():
    # Studies A (3 rows), B (1 row), C (5 rows); index intentionally non-trivial.
    studies = ["A", "A", "A", "B", "C", "C", "C", "C", "C"]
    return pd.DataFrame({"study": studies, "age": range(len(studies))})


def test_filter_studies_none_keeps_all_and_resets_index():
    df = _study_frame().iloc[::-1]  # shuffle the index
    out, dropped = data_utils.filter_studies_by_min_obs(df, None)
    assert dropped == []
    assert len(out) == len(df)
    assert list(out.index) == list(range(len(df)))  # index reset


def test_filter_studies_zero_is_noop():
    df = _study_frame()
    out, dropped = data_utils.filter_studies_by_min_obs(df, 0)
    assert dropped == []
    assert len(out) == len(df)


def test_filter_studies_drops_small_studies():
    df = _study_frame()
    out, dropped = data_utils.filter_studies_by_min_obs(df, 3)
    # B has a single observation -> dropped; A (3) and C (5) retained.
    assert dropped == ["B"]
    assert set(out["study"]) == {"A", "C"}
    assert len(out) == 8
    assert list(out.index) == list(range(8))  # index reset


def test_filter_studies_dropped_list_is_sorted():
    df = pd.DataFrame({"study": ["z", "a", "a", "m"], "age": [1, 2, 3, 4]})
    _out, dropped = data_utils.filter_studies_by_min_obs(df, 2)
    assert dropped == ["m", "z"]  # both singletons, sorted


# Minimal schemas for the per-study source tables referenced by the
# vocab_combined view. The DS regression tests populate wordbank_child only,
# so these stay empty; they exist so the view binds.
_SOURCE_TABLE_SCHEMAS = {
    "vocab_uk_01": "subject_id VARCHAR, sex VARCHAR, age DOUBLE, understood INTEGER, spoken INTEGER, signed INTEGER, produced INTEGER, survey_vocab_max INTEGER",
    "vocab_uk_02": "subject_id VARCHAR, gender INTEGER, age DOUBLE, comprehension INTEGER, spoken INTEGER, signed INTEGER, production INTEGER, form VARCHAR",
    "vocab_ie_01": "subject_id VARCHAR, age_months_start DOUBLE, understands_total_start INTEGER, says_total_start INTEGER, age_months_end DOUBLE, understands_total_end INTEGER, says_total_end INTEGER",
    "vocab_uk_03": "subject_id VARCHAR, age DOUBLE, comprehension INTEGER, production INTEGER",
    "vocab_it_01": "subject_id VARCHAR, age DOUBLE, understood INTEGER, spoken INTEGER, form_max_spoken INTEGER",
    "vocab_uk_04": "subject_id VARCHAR, age DOUBLE, understood INTEGER, spoken INTEGER, signed INTEGER",
    "vocab_uk_05": "subject_id VARCHAR, age DOUBLE, understood INTEGER, spoken INTEGER, signed INTEGER",
    "vocab_us_02": "subject_id VARCHAR, age DOUBLE, understood INTEGER, spoken INTEGER",
    "vocab_uk_06": "subject_id VARCHAR, age DOUBLE, understood INTEGER, spoken INTEGER, signed INTEGER",
    "vocab_ie_02": "subject_id VARCHAR, age DOUBLE, understood INTEGER, spoken INTEGER, signed INTEGER, english_speaking VARCHAR",
    "vocab_nz_01": "subject_id VARCHAR, age BIGINT, not_spoken_or_signed BIGINT, signed BIGINT, spoken_signed BIGINT, spoken BIGINT",
}


def _create_vocab_db(tmp_path):
    db_path = tmp_path / "vocabulary.duckdb"
    with duckdb.connect(str(db_path)) as con:
        con.execute(
            """
            CREATE TABLE wordbank_child (
                form VARCHAR,
                language VARCHAR,
                dataset_name VARCHAR,
                child_id VARCHAR,
                sex VARCHAR,
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
            INSERT INTO wordbank_child VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                ("WG",         "English (British)",  "Fenson (2007)",   "c01", None, 12.0,  40,  12, True,  None),
                ("Oxford CDI", "English (British)",  "Hamilton (2000)", "c02", None, 24.0, 190,  80, True,  None),
                ("WS",         "English (British)",  "Fenson (2007)",   "c03", None, 28.0,  51,  51, True,  None),
                ("WSShort",    "English (British)",  "Fenson (2007)",   "c04", None, 22.0,  20,  20, True,  None),
                ("TEDS Twos",  "English (British)",  "Fenson (2007)",   "c05", None, 24.0,  75,  75, True,  None),
                ("WG",         "English (British)",  "Fenson (2007)",   "c06", None, 35.0, 100,  50, True,  None),
                ("WG",         "English (British)",  "Fenson (2007)",   "c07", None, 18.0,  60,  20, True,  "premature"),
                ("WG",         "English (British)",  "Fenson (2007)",   "c08", None, 18.0,  60,  20, False, None),
                # Non-English row that otherwise matches: excluded by the default English filter.
                ("WG",         "Norwegian",           "Simonsen (2014)", "c09", None, 16.0,  55,  18, True,  None),
                # us_01 / Edgin Down syndrome rows, feeding the vocab_combined
                # view: WG comprehension is an independent measure; WS
                # comprehension is a production proxy and must not become
                # `understood`; production > 100 rows are excluded by the
                # legacy cap in the view.
                ("WG",         "English (American)",  "Edgin",           "d01", "F",  14.0,  40,  12, False, "Down syndrome"),
                ("WS",         "English (American)",  "Edgin",           "d02", "M",  24.0,  77,  77, False, "Down syndrome"),
                ("WS",         "English (American)",  "Edgin",           "d03", None, 29.0, 150, 150, False, "Down syndrome"),
                ("WG",         "English (American)",  "Edgin",           "d04", "F",  18.0, 200, 120, False, "Down syndrome"),
            ],
        )
        for table, table_schema in _SOURCE_TABLE_SCHEMAS.items():
            con.execute(f"CREATE TABLE {table} ({table_schema})")
        con.execute(data_utils.vocab_combined_view_sql())
    return db_path
