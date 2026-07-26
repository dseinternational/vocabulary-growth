# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

import duckdb
import numpy as np
import pandas as pd
import pytest

import vocab_growth.data_utils as data_utils
from vocab_growth.models.definitions import Population


def test_mask_incomparable_signed_outcomes_preserves_other_outcomes():
    frame = pd.DataFrame(
        {
            "study": ["uk_01", "uk_06", "uk_04"],
            "understood": [100, 110, 120],
            "spoken": [40, 50, 60],
            "signed": [20, 30, 40],
        }
    )

    masked, dropped = data_utils.mask_incomparable_signed_outcomes(frame)

    assert dropped == {"uk_01": 1, "uk_06": 1}
    assert masked["understood"].tolist() == frame["understood"].tolist()
    assert masked["spoken"].tolist() == frame["spoken"].tolist()
    assert np.isnan(masked.loc[0, "signed"])
    assert np.isnan(masked.loc[1, "signed"])
    assert masked.loc[2, "signed"] == 40
    assert frame["signed"].tolist() == [20, 30, 40]


def test_mask_incomparable_signed_outcomes_can_reinclude_sources():
    frame = pd.DataFrame(
        {"study": ["uk_01", "uk_06"], "signed": [20, 30]}
    )

    kept, dropped = data_utils.mask_incomparable_signed_outcomes(
        frame,
        include_signed_only=True,
        include_uncertain=True,
    )

    assert dropped == {}
    assert kept["signed"].tolist() == [20, 30]


def test_select_one_observation_per_subject_is_reproducible_and_study_scoped():
    frame = pd.DataFrame(
        {
            "study": ["A", "A", "A", "B", "B"],
            "subject_id": ["1", "1", "2", "1", "1"],
            "age": [10, 12, 11, 9, 13],
        }
    )

    selected_a = data_utils.select_one_observation_per_subject(
        frame, random_seed=47
    )
    selected_b = data_utils.select_one_observation_per_subject(
        frame, random_seed=47
    )

    pd.testing.assert_frame_equal(selected_a, selected_b)
    assert len(selected_a) == 3
    assert not selected_a.duplicated(["study", "subject_id"]).any()


def test_select_one_observation_requires_unique_index():
    frame = pd.DataFrame(
        {
            "study": ["A", "A"],
            "subject_id": ["1", "1"],
            "age": [10, 12],
        },
        index=[0, 0],
    )

    with pytest.raises(ValueError, match="unique dataframe index"):
        data_utils.select_one_observation_per_subject(frame, random_seed=47)


@pytest.mark.parametrize("subject_id", [None, np.nan, "", "   "])
def test_validate_subject_ids_rejects_missing_or_blank_values(subject_id):
    frame = pd.DataFrame({"study": ["A"], "subject_id": [subject_id]})

    with pytest.raises(ValueError, match="non-missing subject ID"):
        data_utils.validate_subject_ids(frame)


def test_us01_ceiling_sensitivity_excludes_only_ws_ceiling_rows():
    frame = pd.DataFrame(
        {
            "study": ["us_01", "us_01", "us_01", "uk_04"],
            "spoken": [680, 679, 396, 416],
            "survey_vocab_max": [680, 680, 396, 416],
        }
    )

    filtered, count = data_utils.exclude_us01_spoken_ceiling_rows(frame)

    assert count == 1
    assert filtered[["study", "spoken"]].to_records(index=False).tolist() == [
        ("us_01", 679),
        ("us_01", 396),
        ("uk_04", 416),
    ]


def test_us01_ceiling_sensitivity_runs_through_ds_loader(tmp_path, monkeypatch):
    db_path = _create_vocab_db(tmp_path)
    with duckdb.connect(str(db_path)) as con:
        con.executemany(
            "INSERT INTO wordbank_child VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    "WS",
                    "English (American)",
                    "Edgin",
                    "d05",
                    None,
                    30,
                    680,
                    680,
                    False,
                    "Down syndrome",
                ),
                (
                    "WG",
                    "English (American)",
                    "Edgin",
                    "d06",
                    None,
                    20,
                    396,
                    396,
                    False,
                    "Down syndrome",
                ),
            ],
        )
    monkeypatch.setattr(data_utils, "VOCABULARY_DATA_PATH", str(db_path))
    frame = data_utils.load_data(
        Population.DOWN_SYNDROME,
        columns=["study", "spoken", "survey_vocab_max"],
    )

    filtered, count = data_utils.exclude_us01_spoken_ceiling_rows(frame)

    assert count == 1
    assert 680 not in filtered["spoken"].tolist()
    assert 396 in filtered["spoken"].tolist()


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


def test_ds_us01_keeps_valid_rows_above_legacy_100_cap(tmp_path, monkeypatch):
    us01 = _load_us01(tmp_path, monkeypatch)

    assert sorted(us01["spoken"].tolist()) == [12, 77, 120, 150]
    assert set(us01["survey_vocab_max"].dropna().astype(int)) == {396, 680}


def test_form_ceiling_guard_drops_counts_above_survey_vocab_max(tmp_path, monkeypatch):
    # issue #131: a count above its source form's native ceiling is impossible
    # (a data-entry error) and must be dropped. it_01 carries a per-row ceiling
    # (form_max_spoken); a row with understood=461 on a 408-item form is
    # excluded, while a count at the ceiling is a legitimate observation kept.
    db_path = _create_vocab_db(tmp_path)
    with duckdb.connect(str(db_path)) as con:
        con.executemany(
            "INSERT INTO vocab_it_01 VALUES (?, ?, ?, ?, ?)",
            [
                ("it_ok", 30.0, 408, 100, 408),  # understood == ceiling: kept
                ("it_bad", 47.0, 461, 12, 408),  # understood 461 > 408: dropped
            ],
        )
    monkeypatch.setattr(data_utils, "VOCABULARY_DATA_PATH", str(db_path))

    it01 = data_utils.load_combined_data()
    it01 = it01[it01["study"] == "it_01"]

    assert it01["understood"].tolist() == [408]  # only the valid (at-ceiling) row survives
    assert 461 not in it01["understood"].tolist()


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
                # `understood`; valid production > 100 rows remain included.
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


# ---- ie_01: comprehension pass-through and the partial baseline wave ----
#
# The view previously set ie_01 understood = GREATEST(says_total, understands_total),
# which repaired away the records where a parent reported saying more words than
# understanding — hiding them from the violation count the nested likelihood
# reports, and feeding it exact spoken == understood rows. And the baseline wave
# omitted DSE Checklist 3, so its counts are on a 460-item frame, not the 810-item
# reference scale. These pin both.

def _ie01_db(tmp_path, rows):
    """A vocabulary DB whose only DS source rows are the supplied ie_01 records."""
    db_path = _create_vocab_db(tmp_path)
    with duckdb.connect(str(db_path)) as con:
        con.executemany(
            "INSERT INTO vocab_ie_01 (subject_id, age_months_start, "
            "understands_total_start, says_total_start, age_months_end, "
            "understands_total_end, says_total_end) VALUES (?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
    return db_path


def test_ie01_understood_is_not_repaired_from_production(tmp_path, monkeypatch):
    # A child reported as saying more than it understands at follow-up. The old
    # GREATEST repair turned this into understood == spoken == 366 (an exact
    # q = 1 observation); understood must now be the parent-reported 13.
    db_path = _ie01_db(tmp_path, [("s1", 30.0, 100, 20, 61.0, 13, 366)])
    monkeypatch.setattr(data_utils, "VOCABULARY_DATA_PATH", str(db_path))

    ie = data_utils.load_combined_data(include_incomplete_administrations=True)
    ie = ie[ie["study"] == "ie_01"].sort_values("age")

    follow_up = ie[ie["age"] == 61.0].iloc[0]
    assert follow_up["understood"] == 13      # not GREATEST(13, 366)
    assert follow_up["spoken"] == 366
    # It is now visible as a source-data violation rather than silently repaired.
    assert (ie["spoken"] > ie["understood"]).sum() == 1


def test_ie01_baseline_wave_carries_its_true_460_item_ceiling(tmp_path, monkeypatch):
    db_path = _ie01_db(tmp_path, [("s1", 30.0, 300, 100, 42.0, 500, 200)])
    monkeypatch.setattr(data_utils, "VOCABULARY_DATA_PATH", str(db_path))

    ie = data_utils.load_combined_data(include_incomplete_administrations=True)
    ie = ie[ie["study"] == "ie_01"].sort_values("age")

    assert list(ie["survey_vocab_max"]) == [460, 810]


def test_ie01_baseline_counts_are_masked_by_default(tmp_path, monkeypatch):
    db_path = _ie01_db(tmp_path, [("s1", 30.0, 300, 100, 42.0, 500, 200)])
    monkeypatch.setattr(data_utils, "VOCABULARY_DATA_PATH", str(db_path))

    masked = data_utils.load_combined_data()
    baseline = masked[(masked["study"] == "ie_01") & (masked["age"] == 30.0)].iloc[0]
    follow_up = masked[(masked["study"] == "ie_01") & (masked["age"] == 42.0)].iloc[0]

    # The partial administration's counts are masked; the row is retained so age
    # coverage stays auditable, and the complete wave is untouched.
    assert pd.isna(baseline["understood"]) and pd.isna(baseline["spoken"])
    assert follow_up["understood"] == 500 and follow_up["spoken"] == 200

    restored = data_utils.load_combined_data(include_incomplete_administrations=True)
    restored_baseline = restored[
        (restored["study"] == "ie_01") & (restored["age"] == 30.0)
    ].iloc[0]
    assert restored_baseline["understood"] == 300


def test_mask_incomplete_administrations_reports_counts_and_needs_columns():
    frame = pd.DataFrame({
        "study": ["ie_01", "ie_01", "uk_03"],
        "survey_vocab_max": [460, 810, 416],
        "understood": [300.0, 500.0, 90.0],
        "spoken": [100.0, 200.0, 30.0],
    })
    out, dropped = data_utils.mask_incomplete_administrations(frame)
    assert dropped == {"ie_01": 2}                      # understood + spoken masked
    assert out["understood"].tolist()[1:] == [500.0, 90.0]
    assert pd.isna(out.loc[0, "understood"])
    # A nested short form is NOT a partial administration: uk_03's 416-item Oxford
    # CDI is left alone (see INCOMPLETE_ADMINISTRATION_CEILINGS).
    assert out.loc[2, "understood"] == 90.0

    out_kept, dropped_kept = data_utils.mask_incomplete_administrations(
        frame, include_incomplete=True
    )
    assert dropped_kept == {}
    assert out_kept.equals(frame)

    with pytest.raises(KeyError, match="survey_vocab_max"):
        data_utils.mask_incomplete_administrations(frame.drop(columns="survey_vocab_max"))
