# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

import os

import duckdb
import numpy as np
import pandas as pd
import pytest

import vocab_growth.data_utils as data_utils
from vocab_growth.models.definitions import Population

requires_real_db = pytest.mark.skipif(
    not os.path.exists(data_utils.VOCABULARY_DATA_PATH),
    reason="prepared vocabulary DuckDB not available (run scripts/prepare_data.py)",
)
"""Skip a test that asserts against the real database rather than a fixture.

``data/vocabulary.duckdb`` is a build artefact, gitignored and rebuilt by
``scripts/prepare_data.py`` in ~300 ms. CI now builds it *before* pytest, so these
tests run there; the marker is for a local checkout where it has not been built
yet. The counts they pin — 8 understood values masked by the duplicated-outcome
rule, 19 spoken by the production rules, every exclusion inside ``us_01`` — are
what the notes and the report quote, and no miniature fixture can check them."""


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

    # uk_01 alone: its `signed` is a sign-ONLY count. uk_06 was masked here until
    # 2026-08-12, when the source confirmed the standard DSE checklists — column 2
    # is "understands and signs", a total — so it is no longer masked.
    assert dropped == {"uk_01": 1}
    assert masked["understood"].tolist() == frame["understood"].tolist()
    assert masked["spoken"].tolist() == frame["spoken"].tolist()
    assert np.isnan(masked.loc[0, "signed"])
    assert masked.loc[1, "signed"] == 30       # uk_06 retained
    assert masked.loc[2, "signed"] == 40
    assert frame["signed"].tolist() == [20, 30, 40]


def test_uncertain_sign_studies_is_empty_but_the_mechanism_survives():
    """uk_06 left the list on evidence; the guard stays for the next source.

    Emptying the tuple rather than deleting it keeps the route open for a future
    source whose signing construct is unverified, and records that this one was
    resolved rather than quietly dropped (issue #211).
    """
    assert data_utils.UNCERTAIN_SIGN_STUDIES == ()

    # The report lists every *excluded* study, so an empty tuple means no study is
    # reported under the uncertain heading at all.
    frame = pd.DataFrame({"study": ["uk_06"], "signed": [30]})
    kept, dropped = data_utils.mask_incomparable_signed_outcomes(frame)
    assert "uk_06" not in dropped
    assert kept.loc[0, "signed"] == 30


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


def test_dse_native_restriction_keeps_only_the_810_reference_form():
    """Native means the form's own ceiling IS the reference, not merely close.

    uk_02 ran both instruments, so the filter has to work per row rather than per
    study, and a row whose ceiling was never recorded has to go: an unknown form
    cannot be shown to need no harmonisation, which is the only thing the variant
    admits rows on.
    """
    frame = pd.DataFrame(
        {
            "study": ["ie_02", "uk_02", "uk_02", "uk_04", "es_01", "uk_03"],
            "survey_vocab_max": [810, 810, 416, 416, 651, None],
        }
    )

    filtered, dropped = data_utils.restrict_to_dse_native_administrations(frame)

    assert dropped == 4
    assert filtered["study"].tolist() == ["ie_02", "uk_02"]
    assert (filtered["survey_vocab_max"] == data_utils.DSE_NATIVE_VOCAB_MAX).all()


@requires_real_db
def test_dse_native_restriction_on_the_real_pool():
    """The real sources: which studies survive, and how much of each outcome.

    Pinned because the variant's whole value is the size of what it removes — if a
    later source arrives on the 810 reference, or an existing one is re-coded, the
    check silently becomes a different check and the numbers quoted in the flag
    docstring stop being true.
    """
    pool = data_utils.load_data(
        population=Population.DOWN_SYNDROME,
        columns=[
            "study", "age", "understood", "spoken", "signed",
            "survey_vocab_max", "subject_id",
        ],
    )
    native, dropped = data_utils.restrict_to_dse_native_administrations(pool)

    assert len(native) == 277
    assert dropped == len(pool) - 277
    assert sorted(native["study"].unique()) == ["ie_01", "ie_02", "uk_02", "uk_06"]
    assert native["subject_id"].nunique() == 194
    # 251, not 259: seven of the ten counts mask_comprehension_below_production
    # masks are ie_01 rows inside this subset, and the withheld ie_02 t2
    # administration (IE02_WITHHELD_ADMINISTRATIONS) took one more.
    assert int(native["understood"].notna().sum()) == 251
    assert int(native["spoken"].notna().sum()) == 263
    assert int(native["signed"].notna().sum()) == 217


def test_us01_ceiling_sensitivity_runs_through_ds_loader(tmp_path, monkeypatch):
    db_path = _create_vocab_db(tmp_path)
    with duckdb.connect(str(db_path)) as con:
        con.executemany(
            f"INSERT INTO vocab_us_01 ({_US01_COLUMNS}) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                # WS at its 680 ceiling, age 30 — the last month inside the WS
                # window, so the window rule keeps it and the implausible-production
                # rule is what masks it.
                ("d05", "WS", 30.0, None, "down_syndrome", 680, 680, 680, True),
                # WG at its 396 ceiling, age 20. Its child has no other record, so
                # the ceiling-only rule removes it before the production rules see it;
                # the reinstatement below therefore has to lift that rule as well.
                ("d06", "WG", 20.0, None, "down_syndrome", 396, 396, 396, False),
            ],
        )
    monkeypatch.setattr(data_utils, "VOCABULARY_DATA_PATH", str(db_path))

    # Neither survives the default loader — the WS record's child has an in-window
    # record so it stays but its count is masked by the implausible-production
    # signature, and the WG record's child is ceiling-only so the whole child is
    # dropped first — so the ceiling helper has nothing to exclude.
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
    reinstated = data_utils.load_combined_data(
        include_implausible_production=True, include_ceiling_only_children=True
    )
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


def test_td_loader_deduplicates_on_complete_source_identity(tmp_path, monkeypatch):
    """Exact full-row copies collapse; rows that collide only after the outcome
    projection survive (#240).

    The Wordbank export records some administrations twice, identically; a
    repeated row double-weights that administration in every likelihood and, in
    the random-effect models, makes a single-visit child look like a
    repeated-measures one. Deduplication must run on the complete source row,
    before the loader's projection: two genuinely distinct same-child, same-age
    administrations can agree on every projected column and differ only in one
    the loader drops (``sex`` here; ``caregiver_education`` and others in the
    real export), and collapsing those would delete a real observation.
    """
    db_path = _create_vocab_db(tmp_path)
    with duckdb.connect(str(db_path)) as con:
        con.executemany(
            "INSERT INTO wordbank_child VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                # An exact copy of the fixture's c01 row: one of the two must go.
                ("WG", "English (British)", "Fenson (2007)", "c01", None, 12.0, 40, 12, True, None),
                # Two distinct administrations that collide after projection.
                ("WG", "English (British)", "Fenson (2007)", "c90", "Female", 14.0, 80, 25, True, None),
                ("WG", "English (British)", "Fenson (2007)", "c90", "Male", 14.0, 80, 25, True, None),
            ],
        )
    monkeypatch.setattr(data_utils, "VOCABULARY_DATA_PATH", str(db_path))

    df = data_utils.load_data(
        Population.TYPICALLY_DEVELOPING,
        columns=["study", "subject_id", "age", "understood", "spoken"],
    )

    exact_copy = df[(df["age"] == 12.0) & (df["understood"] == 40)]
    assert len(exact_copy) == 1
    projection_collision = df[(df["age"] == 14.0) & (df["understood"] == 80)]
    assert len(projection_collision) == 2


@requires_real_db
def test_td_frames_of_record_are_free_of_source_duplicates():
    """Pin the registered VG11-VG13 frame sizes after source-level dedup (#240).

    Before the loader deduplicated, 22 excess exact source copies entered
    VG11's frame (18,522 rows), 3 VG12's (7,052) and 2 VG13's (6,358); the
    removal counts of record are the differences pinned here. These pins move
    only when the export is refreshed — re-run the audit in
    notes/202608231830-vg11-vg13-immediate-remediation.md when they do.
    """
    from vocab_growth.models.definitions import VG11, VG12, VG13

    def univariate_frame_len(definition):
        y_col = definition.outcome.value
        df = data_utils.load_data(
            definition.population,
            ["age", y_col, "study", "subject_id"],
            languages=definition.td_languages,
        )
        df = df.dropna(subset=["age", y_col])
        df, _ = data_utils.filter_studies_by_min_obs(
            df, definition.min_study_observations
        )
        return len(df)

    assert univariate_frame_len(VG11) == 18_500  # was 18,522
    assert univariate_frame_len(VG12) == 7_049  # was 7,052

    df = data_utils.load_data(
        VG13.population,
        ["age", "understood", "spoken", "study", "subject_id"],
        languages=VG13.td_languages,
        max_age_months=VG13.max_age_months,
    )
    df = df.dropna(subset=["age"])
    df = df[df["understood"].notna() | df["spoken"].notna()]
    df, _ = data_utils.filter_studies_by_min_obs(df, VG13.min_study_observations)
    assert len(df) == 6_356  # was 6,358


def test_td_romance_scope_admits_italian_and_spanish_only(tmp_path, monkeypatch):
    """ENGLISH_AND_ROMANCE_LANGUAGES widens the pool to exactly two more languages.

    The point of the widening is DS-TD language symmetry: the DS pool is already a
    quarter non-English (es_01 Spanish, it_01 Italian) while this reference was
    English-only. Norwegian stands in for every language that is *not* admitted --
    widening must not become "all languages", which would change what the reference
    trajectory means and admit forms whose comprehension is a production proxy.
    """
    db_path = _create_vocab_db(tmp_path)
    monkeypatch.setattr(data_utils, "VOCABULARY_DATA_PATH", str(db_path))

    widened = data_utils.load_data(
        Population.TYPICALLY_DEVELOPING,
        columns=["language", "form", "age", "understood"],
        languages=data_utils.ENGLISH_AND_ROMANCE_LANGUAGES,
    )
    languages = set(widened["language"])

    assert {"Italian", "Spanish (European)"} <= languages
    assert "Norwegian" not in languages

    english_only = data_utils.load_data(
        Population.TYPICALLY_DEVELOPING,
        columns=["language", "form", "age", "understood"],
    )
    assert not {"Italian", "Spanish (European)"} & set(english_only["language"])
    assert len(widened) == len(english_only) + 2


def test_td_pool_stays_inside_the_gp_domain_when_languages_widen(tmp_path, monkeypatch):
    """The reference pool must not reach below the typically-developing GP domain.

    Italian Words & Gestures is registered from 7 months where every English form
    starts at 8, so widening the language scope pushed five administrations below the
    floor of ``_TD_GP_DOMAIN_MONTHS`` and ``build_utils`` refused to build. Bounding
    the pool is the fix rather than widening that domain, which is shared with
    VG03/VG04 and would have made those models stale for five observations at the
    least informative end of the range.
    """
    from vocab_growth.models import definitions

    db_path = _create_vocab_db(tmp_path)
    with duckdb.connect(str(db_path)) as con:
        con.execute(
            "INSERT INTO wordbank_child VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("WG", "Italian", "Caselli", "c12", None, 7.0, 9, 0, True, None),
        )
    monkeypatch.setattr(data_utils, "VOCABULARY_DATA_PATH", str(db_path))

    df = data_utils.load_data(
        Population.TYPICALLY_DEVELOPING,
        columns=["language", "age", "understood"],
        languages=data_utils.ENGLISH_AND_ROMANCE_LANGUAGES,
    )

    age_lower, age_upper = data_utils.TD_POOL_AGE_MONTHS
    assert df["age"].min() >= age_lower
    assert 7.0 not in df["age"].tolist()
    # The bound has to sit inside the GP domain the TD models declare, or the pool
    # can still produce ages build_utils will reject.
    domain_low, domain_high = definitions.VG12.gp_domain_months
    assert age_lower >= domain_low
    assert age_upper <= domain_high


def test_romance_scope_is_a_superset_of_english_and_excludes_french():
    """French is excluded on measurement grounds, not by oversight.

    Its Words & Gestures form carries 713 word items where every other Words &
    Gestures adaptation has 309-457, and 20.9% of its rows with comprehension >= 20
    record comprehension exactly equal to production -- the proxy-defect signature
    that retired VG06. A future widening that reaches for "the Romance languages"
    should have to notice this.
    """
    assert set(data_utils.ENGLISH_LANGUAGES) < set(
        data_utils.ENGLISH_AND_ROMANCE_LANGUAGES
    )
    assert set(data_utils.ROMANCE_LANGUAGES) == {"Italian", "Spanish (European)"}
    assert not any(
        "French" in language for language in data_utils.ENGLISH_AND_ROMANCE_LANGUAGES
    )


def test_only_hierarchical_td_models_go_beyond_english():
    """VG03/VG04 must stay English-only; VG11/VG12/VG13 carry the widened scope.

    VG03 and VG04 have no random effects, so between-language variation would be
    absorbed by the Beta-Binomial dispersion and reported as child-level dispersion.
    The widened scope belongs only where a study random intercept can hold it.
    """
    from vocab_growth.models import definitions

    for name in ("VG03", "VG04"):
        definition = getattr(definitions, name)
        assert definition.td_languages == definitions.ENGLISH_LANGUAGES, name
    for name in ("VG11", "VG12", "VG13"):
        definition = getattr(definitions, name)
        assert (
            definition.td_languages == definitions.ENGLISH_AND_ROMANCE_LANGUAGES
        ), name


def test_ds_models_ignore_the_td_language_scope(tmp_path, monkeypatch):
    """The DS subset is English by construction, so a language scope cannot alter it."""
    db_path = _create_vocab_db(tmp_path)
    monkeypatch.setattr(data_utils, "VOCABULARY_DATA_PATH", str(db_path))

    columns = ["study", "age", "understood", "spoken"]
    default = data_utils.load_data(Population.DOWN_SYNDROME, columns=columns)
    widened = data_utils.load_data(
        Population.DOWN_SYNDROME,
        columns=columns,
        languages=data_utils.ENGLISH_AND_ROMANCE_LANGUAGES,
    )

    pd.testing.assert_frame_equal(default, widened)


def test_td_pool_excludes_the_edgin_clinical_cohort(tmp_path, monkeypatch):
    """The reference pool must not contain rows from the audited DS source.

    Two Edgin rows satisfy the typically-developing filter, and one is a Words &
    Sentences record at exactly the 680-word ceiling inside the run that
    ``mask_implausible_production_administrations`` excludes on the DS side. The
    source team no longer holds the original files, so the defect cannot be
    resolved at source — which makes it the more important that the benchmark and
    the benchmarked are disjoint.
    """
    db_path = _create_vocab_db(tmp_path)
    monkeypatch.setattr(data_utils, "VOCABULARY_DATA_PATH", str(db_path))

    spoken = data_utils.load_data(
        Population.TYPICALLY_DEVELOPING,
        columns=["study", "form", "age", "spoken"],
    )
    bivariate = data_utils.load_data(
        Population.TYPICALLY_DEVELOPING,
        columns=["study", "form", "age", "understood", "spoken"],
    )

    assert "Edgin" not in set(spoken["study"])
    assert "Edgin" not in set(bivariate["study"])
    # The ceiling record specifically, which the spoken-only pool would admit.
    assert 680 not in set(spoken["spoken"])


def _create_vocab_db_with_masked_production(tmp_path):
    """Fixture DB plus one us_01 record the implausible-production rule masks."""
    db_path = _create_vocab_db(tmp_path)
    with duckdb.connect(str(db_path)) as con:
        con.executemany(
            f"INSERT INTO vocab_us_01 ({_US01_COLUMNS}) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                ("d09", "WS", 26.0, None, "down_syndrome", 680, 680, 680, True),
                # The same child, below its form's ceiling. Without this the child is
                # ceiling-only and exclude_ceiling_only_children removes it outright,
                # leaving the production reinstatement nothing to put back — which
                # would test the wrong rule.
                ("d09", "WG", 15.0, None, "down_syndrome", 45, 3, 396, True),
            ],
        )
    return db_path


def test_load_data_passes_reinstatement_flags_through_for_ds(tmp_path, monkeypatch):
    """The sensitivity flag must reach the DS frame, not stop at load_data."""
    db_path = _create_vocab_db_with_masked_production(tmp_path)
    monkeypatch.setattr(data_utils, "VOCABULARY_DATA_PATH", str(db_path))

    columns = ["study", "age", "spoken", "survey_vocab_max"]
    masked = data_utils.load_data(
        Population.DOWN_SYNDROME, columns=columns
    )
    reinstated = data_utils.load_data(
        Population.DOWN_SYNDROME,
        columns=columns,
        include_implausible_production=True,
    )

    assert reinstated["spoken"].notna().sum() == masked["spoken"].notna().sum() + 1
    assert 680 in reinstated["spoken"].dropna().tolist()
    assert 680 not in masked["spoken"].dropna().tolist()


def test_load_data_rejects_reinstatement_flags_for_td(tmp_path, monkeypatch):
    """Each flag names a DS defect class, so a TD caller is mistaken, not a no-op."""
    db_path = _create_vocab_db(tmp_path)
    monkeypatch.setattr(data_utils, "VOCABULARY_DATA_PATH", str(db_path))

    with pytest.raises(ValueError, match="Down syndrome pool only"):
        data_utils.load_data(
            Population.TYPICALLY_DEVELOPING,
            columns=["age", "spoken"],
            include_implausible_production=True,
        )


def test_reinstated_implausible_production_count_is_reported(tmp_path, monkeypatch):
    """The count the sensitivity's fit log prints must be non-zero and exact."""
    db_path = _create_vocab_db_with_masked_production(tmp_path)
    monkeypatch.setattr(data_utils, "VOCABULARY_DATA_PATH", str(db_path))

    count = data_utils.count_reinstated_implausible_production()
    masked = data_utils.load_combined_data()
    reinstated = data_utils.load_combined_data(include_implausible_production=True)

    assert count > 0
    assert count == int(
        reinstated["spoken"].notna().sum() - masked["spoken"].notna().sum()
    )


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


def test_ds_us01_admits_only_the_down_syndrome_group(tmp_path, monkeypatch):
    """The Edgin cohort has four developmental-status groups; only one is us_01.

    The comparison group (source ``DevStatus``/``DevelopmentalDiagnosis`` = 0) reaches
    the export as ``typically_developing = false`` with a *blank* condition label,
    because Wordbank's importer links a HealthCondition whose name is the empty string.
    That makes it look, in the export, like Down syndrome children with a missing code.
    It is not: no child carries both codes. Keying on ``dev_status`` keeps it out of the
    DS relation while leaving it available in ``vocab_us_01`` for a matched analysis.
    """
    us01 = _load_us01(tmp_path, monkeypatch)

    # k01 is the comparison-group row in the fixture: 138 understood at 15 months,
    # well clear of every DS value, so it would be conspicuous if it leaked in.
    assert 138 not in us01["understood"].dropna().tolist()
    assert 43 not in us01["spoken"].dropna().tolist()
    assert len(us01) == 4


def test_above_window_administrations_are_admitted(tmp_path, monkeypatch):
    """An early-vocabulary form given to an older child is admissible.

    For a Down syndrome cohort that is developmentally appropriate, not an error, and
    the registered age window governs whether Wordbank's *percentile norms* apply --
    which this project does not use. Excluding them was also the more biased choice: a
    child still on Words & Gestures at 25 months is plausibly lower-ability than one who
    moved to Words & Sentences, and in the real data these are ``us_01``'s **only**
    comprehension observations between 19 and 27 months.
    """
    db_path = _create_vocab_db(tmp_path)
    with duckdb.connect(str(db_path)) as con:
        con.execute(
            f"INSERT INTO vocab_us_01 ({_US01_COLUMNS}) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            # Words & Gestures at 23 months: above the 8-18 window, ordinary counts,
            # and the child also has an in-window record (d01 is a different child, so
            # give this one two of its own).
            ("d07", "WG", 23.0, None, "down_syndrome", 110, 10, 396, False),
        )
        con.execute(
            f"INSERT INTO vocab_us_01 ({_US01_COLUMNS}) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("d07", "WG", 15.0, None, "down_syndrome", 40, 2, 396, True),
        )
    monkeypatch.setattr(data_utils, "VOCABULARY_DATA_PATH", str(db_path))

    us01 = data_utils.load_combined_data()
    us01 = us01[us01["study"] == "us_01"]

    assert 23.0 in us01["age"].tolist()
    assert 110 in us01["understood"].dropna().tolist()


def test_below_form_floor_administrations_are_dropped(tmp_path, monkeypatch):
    """Administrations below a form's lowest registered age are held back.

    Unlike the above-window case this block is unreliable: in the real data three of
    the 16 rows report 236-368 words *spoken* at 6 months, which no 6-month-old in any
    population produces, and two of the same children show comprehension collapsing
    from 247-371 words at 6 months to 5-19 by 11-12. See FORM_AGE_FLOORS.
    """
    db_path = _create_vocab_db(tmp_path)
    with duckdb.connect(str(db_path)) as con:
        con.executemany(
            f"INSERT INTO vocab_us_01 ({_US01_COLUMNS}) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                # 236 is one of the three real impossible counts and sits below
                # 0.9 * 396, so this row tests the floor rule rather than the
                # ceiling-only rule.
                ("w02", "WG", 6.0, None, "down_syndrome", 247, 236, 396, False),
                ("w03", "WG", 7.0, None, "down_syndrome", 28, 0, 396, False),
            ],
        )
    monkeypatch.setattr(data_utils, "VOCABULARY_DATA_PATH", str(db_path))

    default = data_utils.load_combined_data()
    assert 236 not in default["spoken"].dropna().tolist()
    assert 6.0 not in default["age"].tolist()
    assert 7.0 not in default["age"].tolist()

    reinstated = data_utils.load_combined_data(include_below_form_floor=True)
    assert len(reinstated) == len(default) + 2
    assert 6.0 in reinstated["age"].tolist()


def test_form_floor_rule_leaves_other_studies_alone():
    """Only studies with a registered floor are touched; the rule never guesses."""
    frame = pd.DataFrame(
        {
            "study": ["us_01", "us_01", "uk_02", "es_01", "us_01", "us_01"],
            "age": [14.0, 6.0, 96.0, 71.0, 24.0, 44.0],
            "survey_vocab_max": [396, 396, 810, 651, None, 680],
        }
    )

    kept, dropped = data_utils.exclude_below_form_floor(frame)

    # uk_02 at 96 months and es_01 at 71 have no registered floor; the us_01 row with an
    # unknown ceiling cannot be matched to a form, so it is kept rather than guessed;
    # and the us_01 row at 44 months is *above* its window, which this rule allows.
    assert dropped == {"us_01": 1}
    assert kept["age"].tolist() == [14.0, 96.0, 71.0, 24.0, 44.0]


def test_ceiling_only_children_are_dropped_whole(tmp_path, monkeypatch):
    """A child recorded only at the form ceiling is a preparation artefact.

    This is a provenance criterion, not selection on the outcome: age and count together
    cannot separate the Edgin ceiling batch from a legitimately able older child, so what
    separates them is that the batch children have no non-ceiling record of their own.
    """
    db_path = _create_vocab_db(tmp_path)
    with duckdb.connect(str(db_path)) as con:
        con.executemany(
            f"INSERT INTO vocab_us_01 ({_US01_COLUMNS}) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                # Ceiling-only child: both records at the ceiling, nothing else.
                ("b01", "WS", 44.0, None, "down_syndrome", 680, 680, 680, False),
                ("b02", "WS", 52.0, None, "down_syndrome", 680, 680, 680, False),
                # Same shape of count, but this child has a non-ceiling record too, so
                # the child is kept and only the count is left to the other rules.
                ("g01", "WS", 40.0, None, "down_syndrome", 680, 680, 680, False),
                ("g01", "WG", 17.0, None, "down_syndrome", 60, 4, 396, True),
            ],
        )
    monkeypatch.setattr(data_utils, "VOCABULARY_DATA_PATH", str(db_path))

    default = data_utils.load_combined_data()
    us01 = default[default["study"] == "us_01"]
    ids = set(us01["subject_id"])

    kept_g01 = {
        sid
        for sid in ids
        if 17.0 in us01.loc[us01["subject_id"].eq(sid), "age"].tolist()
        and 40.0 in us01.loc[us01["subject_id"].eq(sid), "age"].tolist()
    }
    assert kept_g01, "a child with a non-ceiling record must survive in full"
    assert 44.0 not in us01["age"].tolist()
    assert 52.0 not in us01["age"].tolist()

    reinstated = data_utils.load_combined_data(include_ceiling_only_children=True)
    assert 44.0 in reinstated["age"].tolist()
    assert len(reinstated) == len(default) + 2


def test_ceiling_only_rule_is_scoped_and_child_level():
    """Scoped to CEILING_ONLY_CHILD_STUDIES, and keyed on study plus subject."""
    frame = pd.DataFrame(
        {
            "study": ["us_01", "us_01", "us_01", "uk_01", "uk_01"],
            "subject_id": ["a", "a", "b", "b", "b"],
            "age": [40.0, 17.0, 44.0, 91.0, 95.0],
            "spoken": [680.0, 4.0, 680.0, 669.0, 660.0],
            "survey_vocab_max": [680, 396, 680, 680, 680],
        }
    )

    kept, dropped = data_utils.exclude_ceiling_only_children(frame)

    # us_01 subject 'a' has a non-ceiling record and survives; 'b' does not and goes.
    # uk_01 subject 'b' is ceiling-only but out of scope — and shares a subject label
    # with us_01's 'b', which must not merge them.
    assert dropped == {"us_01": 1}
    assert kept["age"].tolist() == [40.0, 17.0, 91.0, 95.0]


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
    "vocab_uk_01": "subject_id VARCHAR, sex INTEGER, age DOUBLE, understood INTEGER, spoken INTEGER, signed INTEGER, produced INTEGER, survey_vocab_max INTEGER",
    "vocab_uk_02": "subject_id VARCHAR, sex INTEGER, age DOUBLE, comprehension INTEGER, spoken INTEGER, signed INTEGER, production INTEGER, form VARCHAR",
    "vocab_ie_01": "subject_id VARCHAR, age_months_start DOUBLE, understands_total_start INTEGER, says_total_start INTEGER, age_months_end DOUBLE, understands_total_end INTEGER, says_total_end INTEGER",
    "vocab_uk_03": "subject_id VARCHAR, age DOUBLE, comprehension INTEGER, production INTEGER",
    "vocab_it_01": "subject_id VARCHAR, age DOUBLE, understood INTEGER, spoken INTEGER, form_max_spoken INTEGER",
    "vocab_uk_04": "subject_id VARCHAR, age DOUBLE, understood INTEGER, spoken INTEGER, signed INTEGER",
    "vocab_uk_05": "subject_id VARCHAR, sex INTEGER, age DOUBLE, understood INTEGER, spoken INTEGER, signed INTEGER",
    "vocab_us_02": "subject_id VARCHAR, age DOUBLE, understood INTEGER, spoken INTEGER",
    "vocab_uk_06": "subject_id VARCHAR, sex INTEGER, age DOUBLE, understood INTEGER, spoken INTEGER, signed INTEGER",
    "vocab_ie_02": "subject_id VARCHAR, sex INTEGER, age DOUBLE, understood INTEGER, spoken INTEGER, signed INTEGER, english_speaking VARCHAR",
    "vocab_nz_01": "subject_id VARCHAR, age BIGINT, not_spoken_or_signed BIGINT, signed BIGINT, spoken_signed BIGINT, spoken BIGINT",
    "vocab_es_01": 'subject_id VARCHAR, pair_id INTEGER, "group" VARCHAR, sex INTEGER, age BIGINT, age_days BIGINT, mental_age DOUBLE, mental_age_level INTEGER, understood INTEGER, spoken INTEGER, gestured INTEGER, spoken_or_gestured INTEGER',
    "vocab_uk_07": 'subject_id VARCHAR, "group" VARCHAR, sex INTEGER, timepoint VARCHAR, age BIGINT, understood INTEGER, spoken INTEGER, signed INTEGER, spoken_signed INTEGER, produced INTEGER, survey_vocab_max INTEGER',
    "vocab_us_01": "subject_id VARCHAR, form VARCHAR, age DOUBLE, sex VARCHAR, dev_status VARCHAR, comprehension INTEGER, production INTEGER, survey_vocab_max INTEGER, in_norming_window BOOLEAN",
}

_US01_COLUMNS = (
    "subject_id, form, age, sex, dev_status, comprehension, production, "
    "survey_vocab_max, in_norming_window"
)


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
                # Non-English rows. Norwegian is excluded by every admitted scope;
                # Italian and Spanish (European) are excluded by the English default
                # but admitted by ENGLISH_AND_ROMANCE_LANGUAGES.
                ("WG",         "Norwegian",           "Simonsen (2014)", "c09", None, 16.0,  55,  18, True,  None),
                ("WG",         "Italian",             "Caselli",         "c10", None, 16.0,  58,  19, True,  None),
                ("WG",         "Spanish (European)",  "Karousou",        "c11", None, 14.0,  52,  15, True,  None),
                # Two Edgin rows in the real export carry typically_developing = true
                # with no health condition, so they would otherwise land in the TD
                # reference pool the DS exclusions are benchmarked against. One of
                # them sits at the 680-word WS ceiling inside the flagged run.
                # These stay in wordbank_child: the TD pool still reads the export.
                ("WS",         "English (American)",  "Edgin",           "t01", None, 29.0, 680, 680, True,  None),
                ("WG",         "English (American)",  "Edgin",           "t02", None, 17.0, 306,   7, True,  None),
            ],
        )
        for table, table_schema in _SOURCE_TABLE_SCHEMAS.items():
            con.execute(f"CREATE TABLE {table} ({table_schema})")
        # us_01 / Edgin Down syndrome rows now reach vocab_combined through
        # vocab_us_01, not the by-child export: WG comprehension is an independent
        # measure; WS comprehension is a production proxy and must not become
        # `understood`; valid production > 100 rows remain included. All four sit
        # inside their form's norming window (WG 8-18, WS 16-30) so the
        # window rule leaves them alone.
        con.executemany(
            f"INSERT INTO vocab_us_01 ({_US01_COLUMNS}) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                ("d01", "WG", 14.0, "F",  "down_syndrome",  40,  12, 396, True),
                ("d02", "WS", 24.0, "M",  "down_syndrome",  77,  77, 680, True),
                ("d03", "WS", 29.0, None, "down_syndrome", 150, 150, 680, True),
                ("d04", "WG", 18.0, "F",  "down_syndrome", 200, 120, 396, True),
                # The comparison group in the same cohort. dev_status keeps it out of
                # the DS relation; it is not Down syndrome children with a blank code.
                ("k01", "WG", 15.0, None, "comparison",    138,  43, 396, True),
            ],
        )
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
    # q = 1 observation). That must never come back: the value is either the
    # parent-reported 13 or absent, and never the production count.
    db_path = _ie01_db(tmp_path, [("s1", 30.0, 100, 20, 61.0, 13, 366)])
    monkeypatch.setattr(data_utils, "VOCABULARY_DATA_PATH", str(db_path))

    reinstated = data_utils.load_combined_data(
        include_incomplete_administrations=True,
        include_comprehension_below_production=True,
    )
    reinstated = reinstated[reinstated["study"] == "ie_01"].sort_values("age")
    follow_up = reinstated[reinstated["age"] == 61.0].iloc[0]
    assert follow_up["understood"] == 13      # not GREATEST(13, 366)
    assert follow_up["spoken"] == 366
    # Visible as a source-data violation rather than silently repaired.
    assert (reinstated["spoken"] > reinstated["understood"]).sum() == 1

    # Masked by default since 2026-08-25: an inclusive comprehension field cannot
    # be exceeded by production. The production count is untouched, because it is
    # not the value in question.
    masked = data_utils.load_combined_data(include_incomplete_administrations=True)
    masked = masked[masked["study"] == "ie_01"].sort_values("age")
    follow_up = masked[masked["age"] == 61.0].iloc[0]
    assert pd.isna(follow_up["understood"])
    assert follow_up["spoken"] == 366


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


# ---- es_01: the Down syndrome filter, and the symbolic-gesture lexicon ----
#
# es_01 (Galeote) is the only source carrying a typically developing comparison
# group in the same CSV, so the view must admit its Down syndrome children only.
# Its `gestured` column counts *symbolic* gestures representing specific lexical
# items — a gestural lexicon scored per word — so it is the repository's `signed`
# construct, and a total one (words gestured whether or not also spoken), like
# uk_02 and nz_01. A gestural total exceeding its own union is impossible and must
# be masked rather than fed to the signing models.

_ES01_COLUMNS = (
    "subject_id, pair_id, \"group\", sex, age, age_days, mental_age, "
    "mental_age_level, understood, spoken, gestured, spoken_or_gestured"
)


def _es01_db(tmp_path, rows):
    """A vocabulary DB whose only DS source rows are the supplied es_01 records."""
    db_path = _create_vocab_db(tmp_path)
    with duckdb.connect(str(db_path)) as con:
        con.executemany(
            f"INSERT INTO vocab_es_01 ({_ES01_COLUMNS}) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
    return db_path


def test_es01_admits_only_the_down_syndrome_group(tmp_path, monkeypatch):
    # A matched pair: same pair_id, one DS child and one TD child. Only the DS
    # child may reach the Down syndrome analysis relation.
    db_path = _es01_db(
        tmp_path,
        [
            ("ds1", 1, "DS", 1, 60, 1800, 28.0, 7, 500, 300, 40, 320),
            ("td1", 1, "TD", 1, 28, 840, 28.0, 7, 480, 290, 30, 300),
        ],
    )
    monkeypatch.setattr(data_utils, "VOCABULARY_DATA_PATH", str(db_path))

    es = data_utils.load_combined_data()
    es = es[es["study"] == "es_01"]

    assert list(es["subject_id"]) == ["ds1"]
    assert es.iloc[0]["age"] == 60
    assert es.iloc[0]["sex"] == "M"


def test_es01_symbolic_gestures_are_the_signed_lexicon(tmp_path, monkeypatch):
    # gestured = 40 is a total symbolic-gesture lexicon, so it is `signed`; the
    # source's own spoken-or-gestured union (320) is `produced`, de-duplicated, so
    # it must not be recomputed as spoken + gestured (which would be 340).
    db_path = _es01_db(
        tmp_path, [("ds1", 1, "DS", 2, 60, 1800, 28.0, 7, 500, 300, 40, 320)]
    )
    monkeypatch.setattr(data_utils, "VOCABULARY_DATA_PATH", str(db_path))

    row = data_utils.load_combined_data()
    row = row[row["study"] == "es_01"].iloc[0]

    assert row["signed"] == 40
    assert row["spoken"] == 300
    assert row["understood"] == 500

    # load_combined_data does not select `produced`, so assert it on the view
    # itself — the signing engines read that column directly.
    with duckdb.connect(str(db_path), read_only=True) as con:
        produced, signed = con.execute(
            "SELECT produced, signed FROM vocab_combined WHERE study = 'es_01'"
        ).fetchone()
    assert produced == 320       # the recorded union, not 300 + 40
    assert signed == 40


def test_es01_masks_a_gestural_total_above_its_own_union(tmp_path, monkeypatch):
    # A union cannot be smaller than either of its parts, so gestured > union is an
    # impossible total and is not a usable signed count. One real row matches
    # (1 spoken, 15 gestured, union 11). The other outcomes survive.
    db_path = _es01_db(
        tmp_path,
        [
            ("ok", 1, "DS", 2, 60, 1800, 28.0, 7, 500, 300, 40, 320),
            ("bad", 2, "DS", 1, 24, 720, 12.7, 2, 82, 1, 15, 11),
        ],
    )
    monkeypatch.setattr(data_utils, "VOCABULARY_DATA_PATH", str(db_path))

    es = data_utils.load_combined_data()
    es = es[es["study"] == "es_01"].set_index("subject_id")

    assert es.loc["ok", "signed"] == 40
    assert pd.isna(es.loc["bad", "signed"])
    # Comprehension and oral production are unaffected by the gesture-entry error,
    # and the row is retained rather than dropped.
    assert es.loc["bad", "understood"] == 82
    assert es.loc["bad", "spoken"] == 1


def test_es01_carries_the_651_item_cdi_down_ceiling(tmp_path, monkeypatch):
    # A child at the comprehension ceiling is kept (the guard drops only counts
    # strictly above it); one above the ceiling is impossible and dropped.
    db_path = _es01_db(
        tmp_path,
        [
            ("at_ceiling", 1, "DS", 2, 60, 1800, 28.0, 7, 651, 300, 40, 320),
            ("impossible", 2, "DS", 1, 60, 1800, 28.0, 7, 652, 300, 40, 320),
        ],
    )
    monkeypatch.setattr(data_utils, "VOCABULARY_DATA_PATH", str(db_path))

    es = data_utils.load_combined_data()
    es = es[es["study"] == "es_01"]

    assert list(es["subject_id"]) == ["at_ceiling"]
    assert list(es["survey_vocab_max"]) == [651]


# ---- uk_07: modality-exclusive cells become any-modality marginals ----
#
# uk_07 (PACT-DS) records the three expressive cells separately — says-only,
# signs-only, and both — following the nz_01 convention rather than the uk_01 /
# ie_02 / uk_04 / uk_05 one, where `spoken` and `signed` are already totals.
# Reading its cells as totals would understate both marginals by the overlap, so
# the view must re-derive them. Unlike nz_01 it also carries comprehension.

_UK07_COLUMNS = (
    'subject_id, "group", sex, timepoint, age, understood, spoken, signed, '
    "spoken_signed, produced, survey_vocab_max"
)


def _uk07_db(tmp_path, rows):
    """A vocabulary DB whose only DS source rows are the supplied uk_07 records."""
    db_path = _create_vocab_db(tmp_path)
    with duckdb.connect(str(db_path)) as con:
        con.executemany(
            f"INSERT INTO vocab_uk_07 ({_UK07_COLUMNS}) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
    return db_path


def test_uk07_exclusive_cells_become_any_modality_marginals(tmp_path, monkeypatch):
    # says-only 200, signs-only 30, both 90: the any-modality marginals are
    # spoken = 200 + 90 = 290 and signed = 30 + 90 = 120, and `produced` is the
    # source's own union of all three (320), not spoken + signed.
    db_path = _uk07_db(
        tmp_path,
        [("c1", "control", 2, "t1", 60, 500, 200, 30, 90, 320, 674)],
    )
    monkeypatch.setattr(data_utils, "VOCABULARY_DATA_PATH", str(db_path))

    row = data_utils.load_combined_data()
    row = row[row["study"] == "uk_07"].iloc[0]

    assert row["understood"] == 500
    assert row["spoken"] == 290
    assert row["signed"] == 120
    assert row["sex"] == "F"

    # load_combined_data does not select `produced`, so assert it on the view
    # itself — the signing engines read that column directly.
    with duckdb.connect(str(db_path), read_only=True) as con:
        (produced,) = con.execute(
            "SELECT produced FROM vocab_combined WHERE study = 'uk_07'"
        ).fetchone()
    assert produced == 320       # the recorded union, not 290 + 120


def test_uk07_pools_both_trial_arms_and_keeps_repeat_visits(tmp_path, monkeypatch):
    # The trial arm is a property of the child, not of the measurement, so both
    # arms enter the pool; and a child's three time points are three rows sharing
    # one subject id, which the repeated-measures models cluster on.
    db_path = _uk07_db(
        tmp_path,
        [
            ("c1", "control", 1, "t1", 40, 200, 50, 10, 20, 80, 674),
            ("c1", "control", 1, "t2", 50, 300, 90, 10, 30, 130, 674),
            ("c1", "control", 1, "t3", 55, 350, 120, 5, 40, 165, 674),
            ("c2", "intervention", 2, "t1", 44, 210, 60, 12, 18, 90, 674),
        ],
    )
    monkeypatch.setattr(data_utils, "VOCABULARY_DATA_PATH", str(db_path))

    uk = data_utils.load_combined_data()
    uk = uk[uk["study"] == "uk_07"]

    assert len(uk) == 4
    assert sorted(uk["subject_id"].unique()) == ["c1", "c2"]
    assert sorted(uk.loc[uk["subject_id"] == "c1", "age"]) == [40, 50, 55]


def test_uk07_carries_the_674_item_reading_cdi_ceiling(tmp_path, monkeypatch):
    # A count at the form ceiling is kept; one above it is impossible and dropped.
    db_path = _uk07_db(
        tmp_path,
        [
            ("at_ceiling", "control", 2, "t3", 90, 674, 400, 20, 100, 520, 674),
            ("impossible", "control", 1, "t3", 90, 675, 400, 20, 100, 520, 674),
        ],
    )
    monkeypatch.setattr(data_utils, "VOCABULARY_DATA_PATH", str(db_path))

    uk = data_utils.load_combined_data()
    uk = uk[uk["study"] == "uk_07"]

    assert list(uk["subject_id"]) == ["at_ceiling"]
    assert list(uk["survey_vocab_max"]) == [674]


def test_uk07_signing_is_a_total_and_is_not_masked(tmp_path, monkeypatch):
    # uk_07's re-derived `signed` counts words signed whether or not also spoken,
    # so it is comparable with uk_02/nz_01/es_01 and is not a SIGNED_ONLY_STUDIES
    # or UNCERTAIN_SIGN_STUDIES case — the signing harmonisation must leave it be.
    assert "uk_07" not in data_utils.SIGNED_ONLY_STUDIES
    assert "uk_07" not in data_utils.UNCERTAIN_SIGN_STUDIES

    db_path = _uk07_db(
        tmp_path,
        [("c1", "intervention", 1, "t2", 66, 400, 100, 25, 60, 185, 674)],
    )
    monkeypatch.setattr(data_utils, "VOCABULARY_DATA_PATH", str(db_path))

    uk = data_utils.load_combined_data()
    uk = uk[uk["study"] == "uk_07"]
    masked, dropped = data_utils.mask_incomparable_signed_outcomes(uk)

    assert "uk_07" not in dropped      # only excluded studies are reported
    assert masked.iloc[0]["signed"] == 85


def test_uk07_withheld_administrations_are_dropped_at_csv_load():
    # The withheld row is keyed by (subject_id, timepoint), so a different
    # timepoint for the same child, and the same timepoint for a different child,
    # both survive. Reinstatement is removing the entry from the constant.
    subject, timepoint = data_utils.UK07_WITHHELD_ADMINISTRATIONS[0]
    raw = pd.DataFrame(
        {
            "subject_id": [subject, subject, "other", "other"],
            "timepoint": [timepoint, "t1", timepoint, "t1"],
            "understood": [191, 349, 400, 380],
            "produced": [489, 185, 200, 150],
        }
    )
    out, dropped = data_utils.drop_uk07_withheld_administrations(raw)

    assert dropped == 1
    assert len(out) == 3
    assert not ((out["subject_id"] == subject) & (out["timepoint"] == timepoint)).any()
    assert list(out.index) == [0, 1, 2]                  # index reset

    with pytest.raises(KeyError):
        data_utils.drop_uk07_withheld_administrations(raw.drop(columns=["timepoint"]))


def test_uk07_withheld_row_is_the_only_production_above_comprehension_row():
    # Pins why the row is withheld, against the committed source: it is the sole
    # administration whose production exceeds its comprehension. If a data update
    # changes that, this test says so rather than letting the constant go stale.
    raw = pd.read_csv(os.path.join(data_utils.local_env.DATA_DIR, "vocab_data_uk_07.csv"))
    violations = raw[raw["produced"] > raw["understood"]]

    assert len(violations) == 1
    assert (
        violations.iloc[0]["subject_id"],
        violations.iloc[0]["timepoint"],
    ) == data_utils.UK07_WITHHELD_ADMINISTRATIONS[0]

    kept, _dropped = data_utils.drop_uk07_withheld_administrations(raw)
    assert (kept["produced"] <= kept["understood"]).all()


def test_uk01_withheld_subjects_are_dropped_at_csv_load():
    # Withholding is by whole subject: the defect is the name-derived identifier
    # itself, so every row under the id goes. Reinstatement is removing the id
    # from the constant.
    subject = data_utils.UK01_WITHHELD_SUBJECTS[0]
    raw = pd.DataFrame(
        {
            "subject_id": [subject, subject, "other"],
            "age": [66, 76, 66],
            "spoken": [8, 451, 100],
        }
    )
    out, dropped = data_utils.drop_uk01_withheld_subjects(raw)

    assert dropped == 2
    assert list(out["subject_id"]) == ["other"]
    assert list(out.index) == [0]                        # index reset

    with pytest.raises(KeyError):
        data_utils.drop_uk01_withheld_subjects(raw.drop(columns=["subject_id"]))


def test_uk01_withheld_subject_interleaves_two_modality_profiles():
    # Pins why the id is withheld, against the committed source: sorted by age,
    # its four administrations alternate between a signer who barely speaks
    # (ages 66 and 78) and a speaker who never signs (76 and 88) — the
    # homonym-fusion signature, not a trajectory. If a data update changes
    # this, the constant has gone stale and this test says so.
    raw = pd.read_csv(
        os.path.join(data_utils.local_env.DATA_DIR, "vocab_data_uk_01.csv")
    )
    rows = raw[
        raw["subject_id"] == data_utils.UK01_WITHHELD_SUBJECTS[0]
    ].sort_values("age")

    assert list(rows["age"]) == [66, 76, 78, 88]
    signer = rows.iloc[[0, 2]]
    speaker = rows.iloc[[1, 3]]
    assert (signer["signed"] >= 100).all()
    assert (signer["spoken"] < 50).all()
    assert (speaker["signed"] == 0).all()
    assert (speaker["spoken"] > 400).all()


def test_ie02_withheld_administrations_are_dropped_at_csv_load():
    # The withheld row is keyed by (subject_id, timepoint): the child's other
    # timepoint, and the same timepoint for another child, both survive.
    subject, timepoint = data_utils.IE02_WITHHELD_ADMINISTRATIONS[0]
    raw = pd.DataFrame(
        {
            "subject_id": [subject, subject, "other"],
            "timepoint": [timepoint, "t1", timepoint],
            "understood": [442, 111, 200],
        }
    )
    out, dropped = data_utils.drop_ie02_withheld_administrations(raw)

    assert dropped == 1
    assert len(out) == 2
    assert not ((out["subject_id"] == subject) & (out["timepoint"] == timepoint)).any()
    assert list(out.index) == [0, 1]                     # index reset

    with pytest.raises(KeyError):
        data_utils.drop_ie02_withheld_administrations(raw.drop(columns=["timepoint"]))


def test_ie02_withheld_administration_is_the_contradictory_t2():
    # Pins why the administration is withheld, against the committed source:
    # its t2 asserts a 331-word comprehension surge, a 237-word signing surge
    # and a 96% speech collapse in three months. If a data update changes
    # these counts, the constant has gone stale and this test says so.
    raw = pd.read_csv(
        os.path.join(data_utils.local_env.DATA_DIR, "vocab_data_ie_02.csv")
    )
    subject, timepoint = data_utils.IE02_WITHHELD_ADMINISTRATIONS[0]
    rows = raw[raw["subject_id"] == subject].set_index("timepoint")

    assert list(rows.loc["t2", ["understood", "spoken", "signed"]]) == [442, 3, 301]
    assert list(rows.loc["t1", ["understood", "spoken", "signed"]]) == [111, 72, 64]

    kept, dropped = data_utils.drop_ie02_withheld_administrations(raw)
    assert dropped == 1
    assert (kept["subject_id"] == subject).sum() == 1    # t1 retained


def test_mask_same_day_production_disagreements_masks_larger_side_only():
    frame = pd.DataFrame(
        {
            "study": ["us_01"] * 6 + ["uk_01"] * 2,
            "subject_id": ["a", "a", "b", "b", "c", "c", "d", "d"],
            "age": [23, 23, 23, 23, 17, 17, 50, 50],
            "spoken": [11.0, 385.0, 10.0, 71.0, 0.0, 19.0, 20.0, 300.0],
            "produced": [11.0, 385.0, 10.0, 71.0, 0.0, 19.0, 20.0, 300.0],
        }
    )
    out, dropped = data_utils.mask_same_day_production_disagreements(frame)

    # a: the contradicted larger count is masked, the smaller kept, row retained.
    assert pd.isna(out.loc[1, "spoken"]) and pd.isna(out.loc[1, "produced"])
    assert out.loc[0, "spoken"] == 11
    # b: below the magnitude floor — 10 vs 71 is small-count noise, untouched.
    assert out.loc[3, "spoken"] == 71
    # c: tiny counts, untouched even at an infinite ratio.
    assert out.loc[5, "spoken"] == 19
    # d: out-of-scope study, untouched at any disagreement.
    assert out.loc[7, "spoken"] == 300
    assert dropped == {"us_01": 2}                       # spoken + produced, one row

    reinstated, none_dropped = data_utils.mask_same_day_production_disagreements(
        frame, include_disagreements=True
    )
    assert none_dropped == {}
    assert reinstated["spoken"].notna().all()

    with pytest.raises(KeyError):
        data_utils.mask_same_day_production_disagreements(frame.drop(columns=["age"]))


def test_us01_same_day_disagreement_rule_masks_exactly_two_counts():
    # Pins the rule's net catch on the real pool: reinstating it puts back
    # exactly two us_01 spoken counts, the same-day-contradicted Words &
    # Sentences records of 385 and 406 words at 23 months. Running after the
    # other production rules, its catch is precisely the counts nothing else
    # explains — raw same-day pairs in the ceiling region are already masked
    # by the near-ceiling, collapse and duplicated-outcome signatures.
    masked = data_utils.load_combined_data()
    reinstated = data_utils.load_combined_data(include_same_day_disagreements=True)

    regained = reinstated["spoken"].notna() & ~masked["spoken"].notna()
    rows = reinstated[regained]

    assert sorted(rows["spoken"]) == [385, 406]
    assert set(rows["study"]) == {"us_01"}
    assert set(rows["age"]) == {23}


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


@requires_real_db
def test_load_combined_data_masks_the_edgin_duplicated_outcomes():
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


@requires_real_db
def test_edgin_rules_together_on_the_real_database():
    # End to end: the widened duplicated-outcome rule masks 8 understood values,
    # the production rules mask 21 spoken values (8 duplicated-outcome + 11
    # near-ceiling/collapse + 2 same-day contradictions), and one duplicate row
    # is dropped.
    #
    # 21, not the 30 this pinned before the us_01 source moved to the item-level
    # contributor files. Fewer counts are masked because more records are removed
    # earlier: exclude_ceiling_only_children takes the Edgin ceiling batch out as whole
    # children on its provenance, so those counts never reach the production rules. See
    # CEILING_ONLY_CHILD_STUDIES.
    base = data_utils.load_combined_data(
        include_duplicated_outcomes=True,
        include_implausible_production=True,
        include_same_day_disagreements=True,
    )
    final = data_utils.load_combined_data()
    assert base["understood"].notna().sum() - final["understood"].notna().sum() == 8
    assert base["spoken"].notna().sum() - final["spoken"].notna().sum() == 21

    # Every exclusion is in us_01, and the retained borderline cases survive.
    for column in ("understood", "spoken"):
        lost = base[column].notna().sum() - final[column].notna().sum()
        in_us01 = (
            base[base["study"] == "us_01"][column].notna().sum()
            - final[final["study"] == "us_01"][column].notna().sum()
        )
        assert lost == in_us01
    # The two same-day-contradicted WS counts are gone from the default pool…
    assert ((final["spoken"] == 406) & (final["age"] == 23)).sum() == 0
    assert ((final["spoken"] == 385) & (final["age"] == 23)).sum() == 0
    # …while the retained high-comprehension borderline cases survive.
    assert (final["understood"].between(210, 220) & (final["age"] == 18)).sum() == 2


def _replication_frame() -> pd.DataFrame:
    """Ten children per study, child ``k`` contributing ``1 + k % 3`` rows."""
    rows = []
    for study in ("StudyA", "StudyB"):
        for child in range(10):
            for visit in range(1 + child % 3):
                rows.append(
                    {
                        "study": study,
                        "subject_id": f"id_{child}",
                        "age": 12 + visit,
                        "spoken": 10 * child + visit,
                    }
                )
    return pd.DataFrame(rows)


def test_subsample_subjects_keeps_every_administration_of_a_selected_child():
    frame = _replication_frame()
    out = data_utils._subsample_subjects(frame, 0.5, random_seed=47)

    full_sizes = (frame["study"] + "::" + frame["subject_id"]).value_counts()
    out_sizes = (out["study"] + "::" + out["subject_id"]).value_counts()

    # Every retained child keeps all of its rows — no child is split.
    assert not out_sizes.empty
    for key, n in out_sizes.items():
        assert n == full_sizes[key]
    # And the draw is over children, so the child count is the fraction, not the
    # row count. This is the regression: a row-wise draw would leave ~half of the
    # repeat children with a single row.
    assert out_sizes.size == round(full_sizes.size * 0.5)


def test_subsample_subjects_is_study_scoped():
    # The same subject_id in two studies is two children, matching the
    # subject_key convention in the random-effect engines. Selecting
    # StudyA::id_k must therefore not drag in StudyB::id_k.
    frame = _replication_frame()
    out = data_utils._subsample_subjects(frame, 0.5, random_seed=47)
    keys = set(out["study"] + "::" + out["subject_id"])
    per_study = {
        s: {k.split("::")[1] for k in keys if k.startswith(f"{s}::")}
        for s in ("StudyA", "StudyB")
    }
    assert per_study["StudyA"] and per_study["StudyB"]
    # At least one child id is drawn in one study but not the other, which is
    # only possible if the two are treated as distinct subjects.
    assert per_study["StudyA"] ^ per_study["StudyB"]


def test_subsample_subjects_is_reproducible():
    frame = _replication_frame()
    first = data_utils._subsample_subjects(frame, 0.5, random_seed=47)
    second = data_utils._subsample_subjects(frame, 0.5, random_seed=47)
    other = data_utils._subsample_subjects(frame, 0.5, random_seed=48)
    pd.testing.assert_frame_equal(first, second)
    assert not first.equals(other)


def test_subsample_subjects_does_not_depend_on_input_row_order():
    """The seed alone must determine the draw.

    ``Series.unique`` preserves order of first appearance and ``Series.sample``
    draws by position, so without the sort the selected children would depend on
    the order DuckDB returned rows in — and the loader's query has no ORDER BY.
    """
    frame = _replication_frame()
    shuffled = frame.sample(frac=1.0, random_state=1).reset_index(drop=True)

    def subjects(df):
        return set(df["study"] + "::" + df["subject_id"])

    assert subjects(
        data_utils._subsample_subjects(frame, 0.5, random_seed=47)
    ) == subjects(data_utils._subsample_subjects(shuffled, 0.5, random_seed=47))


@requires_real_db
def test_td_sample_fraction_preserves_within_child_replication():
    """A subsample must not flatten the pool to one administration per child.

    Drawing rows rather than children cut the typically-developing pool from 1.32
    administrations per child to 1.04, which made the subject random intercept
    and the observation-level Beta-Binomial dispersion indistinguishable and gave
    VG11 a bimodal posterior at R-hat 1.72. See
    notes/202608020829-kappa-and-eta-q-prior-recalibration.md §§11-12.
    """
    columns = ["age", "spoken", "study", "subject_id"]

    def obs_per_subject(df):
        df = df.dropna(subset=["age", "spoken"])
        key = df["study"].astype(str) + "::" + df["subject_id"].astype(str)
        return len(df) / key.nunique()

    full = obs_per_subject(
        data_utils.load_data(Population.TYPICALLY_DEVELOPING, columns)
    )
    sampled = obs_per_subject(
        data_utils.load_data(
            Population.TYPICALLY_DEVELOPING, columns, sample_fraction=0.10
        )
    )

    assert full > 1.1, "fixture assumption: the TD pool has repeated measures"
    # Subject-wise draw keeps replication; the row-wise draw it replaced scored
    # 1.04 here against a pool value of 1.32.
    assert sampled == pytest.approx(full, rel=0.10)


# --- Comprehension below production (issue #190 item C; ruling 2026-08-25) ------


def _cbp_frame(rows):
    return pd.DataFrame(
        rows,
        columns=["study", "age", "understood", "spoken", "signed", "produced"],
    )


def test_comprehension_below_production_masks_only_the_comprehension_count():
    frame = _cbp_frame([("ie_01", 61.0, 13.0, 366.0, 0.0, 366.0)])
    out, masked = data_utils.mask_comprehension_below_production(frame)

    # Only `understood` goes. The production figure is corroborated by two
    # columns that agree, and in both diagnosed studies the fault is localised
    # to comprehension.
    assert pd.isna(out.loc[0, "understood"])
    assert out.loc[0, "spoken"] == 366.0
    assert out.loc[0, "produced"] == 366.0
    assert len(out) == 1                      # row retained for provenance
    assert masked == {"ie_01": 1}

    kept, kept_masked = data_utils.mask_comprehension_below_production(
        frame, include_below_production=True
    )
    assert kept_masked == {}
    assert kept.equals(frame)


def test_comprehension_equal_to_production_is_kept():
    # A child who produces everything they understand is legitimate, and so is a
    # child who yet knows nothing. Equality is not a violation.
    frame = _cbp_frame([
        ("us_01", 24.0, 120.0, 120.0, 0.0, 120.0),
        ("us_01", 9.0, 0.0, 0.0, 0.0, 0.0),
    ])
    out, masked = data_utils.mask_comprehension_below_production(frame)
    assert out["understood"].tolist() == [120.0, 0.0]
    assert masked == {}


def test_comprehension_rule_uses_produced_not_the_modality_sum():
    # A bimodal child who says and signs many of the same words: `spoken +
    # signed` overstates distinct production and would flag this row, but the
    # recorded union does not. uk_07 has produced < spoken + signed on 77 of 82
    # rows, so the sum is the wrong denominator, not a conservative one.
    frame = _cbp_frame([("uk_07", 40.0, 300.0, 200.0, 180.0, 290.0)])
    out, masked = data_utils.mask_comprehension_below_production(frame)
    assert out.loc[0, "understood"] == 300.0
    assert masked == {}


def test_comprehension_rule_requires_produced():
    frame = _cbp_frame([("ie_01", 61.0, 13.0, 366.0, 0.0, 366.0)])
    with pytest.raises(KeyError, match="produced"):
        data_utils.mask_comprehension_below_production(frame.drop(columns="produced"))


def test_comprehension_rule_counts_per_study():
    frame = _cbp_frame([
        ("ie_01", 54.0, 33.0, 160.0, 0.0, 160.0),
        ("ie_01", 35.0, 18.0, 27.0, 0.0, 27.0),
        ("uk_01", 38.0, 142.0, 164.0, 0.0, 164.0),
        ("it_01", 59.0, 58.0, 373.0, 0.0, 373.0),
        ("us_01", 24.0, 400.0, 100.0, 0.0, 100.0),   # ordinary: kept
    ])
    _, masked = data_utils.mask_comprehension_below_production(frame)
    assert masked == {"ie_01": 2, "it_01": 1, "uk_01": 1}


@requires_real_db
def test_load_combined_data_masks_the_ten_impossible_comprehension_counts():
    # End-to-end against the real database. Ten administrations across three
    # studies record a comprehension count below the child's own production.
    masked = data_utils.load_combined_data()
    reinstated = data_utils.load_combined_data(
        include_comprehension_below_production=True
    )
    difference = int(
        reinstated["understood"].notna().sum() - masked["understood"].notna().sum()
    )
    assert difference == 10
    # The flag reinstates comprehension only; nothing else moves, and `produced`
    # never reaches a caller.
    assert "produced" not in masked.columns
    assert len(masked) == len(reinstated)
    assert masked["spoken"].notna().sum() == reinstated["spoken"].notna().sum()


def test_the_comprehension_reinstatement_reaches_the_supported_fit_interface():
    """Every documented reinstatement flag must be reachable through `load_data`.

    This one was implemented on `load_combined_data` alone, so the interface
    every model and sensitivity variant actually uses could not request it —
    a documented sensitivity that could not be run (issue #266).
    """
    columns = ["age", "understood", "spoken"]
    masked = data_utils.load_data(
        population=data_utils.Population.DOWN_SYNDROME, columns=columns
    )
    reinstated = data_utils.load_data(
        population=data_utils.Population.DOWN_SYNDROME,
        columns=columns,
        include_comprehension_below_production=True,
    )
    difference = int(
        reinstated["understood"].notna().sum() - masked["understood"].notna().sum()
    )
    assert difference == 10

    # It is a Down-syndrome-pool defect class, so asking for it on the
    # typically-developing pool is a caller error rather than a silent no-op.
    with pytest.raises(ValueError, match="Down syndrome pool only"):
        data_utils.load_data(
            population=data_utils.Population.TYPICALLY_DEVELOPING,
            columns=columns,
            include_comprehension_below_production=True,
        )


def test_produced_is_returned_only_on_request():
    """`produced` is the modality union; no registered model consumes it.

    It is kept out of the default column set so every existing caller sees the
    historical frame, and offered on request so the exploratory produced-outcome
    models can use the canonical loader instead of bypassing it (issue #266).
    """
    default = data_utils.load_combined_data()
    with_produced = data_utils.load_combined_data(include_produced=True)

    assert "produced" not in default.columns
    assert "produced" in with_produced.columns
    assert len(default) == len(with_produced)
    pd.testing.assert_frame_equal(
        with_produced.drop(columns=["produced"]), default
    )


def test_the_prepared_frame_has_a_deterministic_row_order():
    """The loader queries carry no ORDER BY, so the order was the scan order.

    Everything statistical is order-invariant, but the fit manifest records an
    exact hash of the prepared frame precisely so a stale posterior can be told
    from a current one — and a hash over a nondeterministic order cannot be
    recomputed for validation (issue #266 finding 1).
    """
    first = data_utils.load_combined_data()
    second = data_utils.load_combined_data()
    pd.testing.assert_frame_equal(first, second)

    # Canonical, not merely repeatable: a shuffled frame sorts back to it.
    shuffled = first.sample(frac=1.0, random_state=11)
    pd.testing.assert_frame_equal(
        data_utils._deterministic_row_order(shuffled), first
    )
