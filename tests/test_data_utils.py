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

    # Both of these now match the implausible-production signature — the WS record
    # sits at its 680-item ceiling at 30 months and the WG record at its 396-item
    # ceiling at 20 months, and both are inside the young window — so the default
    # loader has already masked them, and the ceiling helper has nothing to exclude.
    # This is why the registered `us01-ceiling-excluded` sensitivities were retired:
    # a check that cannot fail is worse than no check (see registry.py).
    masked = data_utils.load_data(
        Population.DOWN_SYNDROME,
        columns=["study", "spoken", "survey_vocab_max"],
    )
    assert 680 not in masked["spoken"].dropna().tolist()
    assert 396 not in masked["spoken"].dropna().tolist()

    _, count = data_utils.exclude_us01_spoken_ceiling_rows(masked)
    assert count == 0

    # The helper still works on a frame where the records are reinstated, so the
    # exclusion remains available if the masking decision is ever reverted.
    reinstated = data_utils.load_combined_data(include_implausible_production=True)
    reinstated = reinstated[["study", "spoken", "survey_vocab_max"]]
    filtered, count = data_utils.exclude_us01_spoken_ceiling_rows(reinstated)
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


# ---- duplicated-outcome administrations (the us_01/Edgin infant records) ----
#
# An infant recorded as saying nearly every word they understand has an internally
# inconsistent administration: comprehension leading production is the premise of
# the joint models. The rule is age-conditioned rather than study-scoped, because
# the same ratio at older ages is ordinary.

def _dup_frame(rows):
    import pandas as pd

    return pd.DataFrame(
        rows, columns=["study", "age", "understood", "spoken", "survey_vocab_max"]
    )


def test_duplicated_outcome_masks_both_counts_and_keeps_the_row():
    frame = _dup_frame([("us_01", 12.0, 386.0, 385.0, 396)])
    out, dropped = data_utils.mask_duplicated_outcome_administrations(frame)

    # Both counts go: which column was overwritten is unrecoverable from totals,
    # and the production figure is impossible against the independent DS cohort.
    assert pd.isna(out.loc[0, "understood"])
    assert pd.isna(out.loc[0, "spoken"])
    assert len(out) == 1                     # row retained for provenance
    assert dropped == {"us_01": 2}

    kept, kept_dropped = data_utils.mask_duplicated_outcome_administrations(
        frame, include_duplicated=True
    )
    assert kept_dropped == {}
    assert kept.equals(frame)


def test_duplicated_outcome_rule_is_age_conditioned():
    # The identical ratio and count at 40 months is a child who says most of what
    # they understand — ordinary, and must not be masked. 21 of the 27 rows in the
    # real pool matching the ratio/count conditions are of this kind.
    frame = _dup_frame([
        ("us_01", 14.0, 350.0, 348.0, 396),   # infancy: masked
        ("uk_02", 40.0, 350.0, 348.0, 810),   # older: kept
    ])
    out, dropped = data_utils.mask_duplicated_outcome_administrations(frame)
    assert pd.isna(out.loc[0, "understood"])
    assert out.loc[1, "understood"] == 350.0
    assert out.loc[1, "spoken"] == 348.0
    assert dropped == {"us_01": 2}


def test_duplicated_outcome_rule_respects_the_understood_floor():
    # A genuine young record where both counts are small and equal: an infant who
    # understands and says the same handful of words is entirely plausible.
    frame = _dup_frame([("uk_04", 12.0, 8.0, 8.0, 416)])
    out, dropped = data_utils.mask_duplicated_outcome_administrations(frame)
    assert out.loc[0, "understood"] == 8.0
    assert dropped == {}


def test_duplicated_outcome_rule_keeps_a_normal_production_gap():
    # "Group 2": high comprehension for an infant, but an ordinary comprehension-
    # production gap. Retained by decision — clinically unusual, not a defect.
    frame = _dup_frame([("us_01", 18.0, 217.0, 22.0, 396)])
    out, dropped = data_utils.mask_duplicated_outcome_administrations(frame)
    assert out.loc[0, "understood"] == 217.0
    assert out.loc[0, "spoken"] == 22.0
    assert dropped == {}


def test_duplicated_outcome_masking_requires_its_columns():
    frame = _dup_frame([("us_01", 12.0, 386.0, 385.0, 396)])
    with pytest.raises(KeyError, match="age"):
        data_utils.mask_duplicated_outcome_administrations(frame.drop(columns="age"))


def test_load_combined_data_masks_the_edgin_duplicated_outcomes(tmp_path, monkeypatch):
    # End-to-end against the real database. The ratio is 0.75, set from the measured
    # gap (0.86 -> 0.55), so eight administrations match — an earlier 0.9 cut ran
    # through the middle of the cluster and missed two.
    masked = data_utils.load_combined_data()
    kept = data_utils.load_combined_data(include_duplicated_outcomes=True)
    assert kept["understood"].notna().sum() - masked["understood"].notna().sum() == 8

    # Count this rule's own effect on a frame with every mask lifted: applied after
    # the production rules, some of its records already have `spoken` masked, so a
    # sequentially-masked frame understates it.
    unmasked = data_utils.load_combined_data(
        include_duplicated_outcomes=True, include_implausible_production=True
    )
    _, dropped = data_utils.mask_duplicated_outcome_administrations(unmasked)
    assert dropped == {"us_01": 16}    # 8 administrations x understood + spoken


# ---- duplicate administrations, and implausible production ----

def _prod_frame(rows):
    import pandas as pd

    return pd.DataFrame(
        rows,
        columns=["study", "subject_id", "age", "understood", "spoken", "survey_vocab_max"],
    )


def test_drop_duplicate_administrations_collapses_a_repeated_row():
    # us_01 records one 11-month administration twice, identically. A repeated row
    # double-weights the observation and makes a single-visit child look repeated.
    frame = _prod_frame([
        ("us_01", "c1", 11.0, 60.0, 1.0, 396),
        ("us_01", "c1", 11.0, 60.0, 1.0, 396),
        ("us_01", "c1", 17.0, 110.0, 9.0, 396),   # genuine repeat visit: different age
    ])
    out, removed = data_utils.drop_duplicate_administrations(frame)
    assert removed == 1
    assert len(out) == 2
    assert sorted(out["age"]) == [11.0, 17.0]


def test_implausible_production_masks_near_ceiling_in_the_young_window():
    frame = _prod_frame([
        ("us_01", "c1", 24.0, None, 680.0, 680),   # at the WS ceiling: masked
        ("us_01", "c2", 17.0, None, 641.0, 680),   # just below, still >= 0.9: masked
        ("us_01", "c3", 23.0, None, 406.0, 680),   # extreme but no signature: kept
    ])
    out, dropped = data_utils.mask_implausible_production_administrations(frame)
    assert pd.isna(out.loc[0, "spoken"])
    assert pd.isna(out.loc[1, "spoken"])
    assert out.loc[2, "spoken"] == 406.0
    assert dropped == {"us_01": 2}


def test_implausible_production_is_scoped_to_the_young_window():
    # The same near-ceiling count at 40 months is plausible for a child with a
    # large vocabulary and must survive.
    frame = _prod_frame([("uk_06", "c1", 40.0, 700.0, 680.0, 810)])
    out, dropped = data_utils.mask_implausible_production_administrations(frame)
    assert out.loc[0, "spoken"] == 680.0
    assert dropped == {}


def test_implausible_production_masks_a_longitudinal_collapse():
    # 454 words at 18 months against 35 at 24 months: vocabulary does not shrink.
    frame = _prod_frame([
        ("us_01", "c1", 18.0, None, 454.0, 680),
        ("us_01", "c1", 24.0, None, 35.0, 680),
    ])
    out, dropped = data_utils.mask_implausible_production_administrations(frame)
    assert pd.isna(out.loc[0, "spoken"])      # the earlier, contradicted value
    assert out.loc[1, "spoken"] == 35.0       # the plausible later one survives
    assert dropped == {"us_01": 1}


def test_longitudinal_collapse_has_a_floor_so_tiny_counts_do_not_fire():
    # 5 understood words falling to 1 is noise, not a defect; without the floor the
    # 5x rule would flag it.
    frame = _prod_frame([
        ("us_01", "c1", 11.0, 12.0, 5.0, 396),
        ("us_01", "c1", 17.0, 20.0, 1.0, 396),
    ])
    out, dropped = data_utils.mask_implausible_production_administrations(frame)
    assert out.loc[0, "spoken"] == 5.0
    assert dropped == {}


def test_implausible_production_leaves_comprehension_alone():
    # Only the production side is masked here; a Words & Gestures comprehension
    # value is an independent measurement and is handled by its own rule.
    frame = _prod_frame([("us_01", "c1", 18.0, 156.0, 396.0, 396)])
    out, _ = data_utils.mask_implausible_production_administrations(frame)
    assert pd.isna(out.loc[0, "spoken"])
    assert out.loc[0, "understood"] == 156.0


def test_edgin_rules_together_on_the_real_database():
    # End to end: the widened duplicated-outcome rule masks 8 understood values,
    # the production rules mask 30 spoken values, and one duplicate row is dropped.
    base = data_utils.load_combined_data(
        include_duplicated_outcomes=True, include_implausible_production=True
    )
    final = data_utils.load_combined_data()
    assert base["understood"].notna().sum() - final["understood"].notna().sum() == 8
    assert base["spoken"].notna().sum() - final["spoken"].notna().sum() == 30

    # Every exclusion is in us_01, and the retained borderline cases survive.
    for column in ("understood", "spoken"):
        lost = base[column].notna().sum() - final[column].notna().sum()
        in_us01 = (
            base[base["study"] == "us_01"][column].notna().sum()
            - final[final["study"] == "us_01"][column].notna().sum()
        )
        assert lost == in_us01
    assert ((final["spoken"] == 406) & (final["age"] == 23)).sum() == 1
    assert (final["understood"].between(210, 220) & (final["age"] == 18)).sum() == 2
