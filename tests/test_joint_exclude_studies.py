# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""`exclude_studies` on the joint definitions (#297 check 5).

The joint engine assembles its cross-tabulation sources into the same frame as
the merged view's marginals, so the filter has to run on the assembled frame:
run on the merged view it would remove a cross-tab study's marginals and leave
its cells in the composition likelihood. These tests pin that, the empty
default, the refusal of a code that matches nothing, and that study and child
codes are assigned after the rows go.

Built from small fixture CSVs rather than the prepared database, so they run
anywhere. The same behaviour was checked on the real VG25 frame when the field
was added: excluding `uk_07` removes its 82 rows and takes the lag's support
from 191 to 139, and excluding `ie_02` takes it to 148 -- the 52 and 43
supporting observations those studies contribute.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

import vocab_growth.environment as env
from vocab_growth.models import common_joint_modality as cjm
from vocab_growth.models.cross_lag import prev_wave_sign_share_lag_for_frame
from vocab_growth.models.definitions import VG15, VG25
from vocab_growth.models.observation_arrays import prepare_joint_observations


def _write_sources(directory) -> None:
    """The four cross-tab CSVs the joint frame builder reads for VG15."""
    pd.DataFrame(
        [
            # A reconciling four-cell row, which feeds the composition...
            dict(subject_id="uk_a", age=30.0, comprehension=19, signed=6, spoken=7,
                 understood_only=10, signed_only=2, spoken_only=3, signed_spoken=4,
                 form="DSE"),
            # ...and a row missing a cell, which is routed to the marginals.
            dict(subject_id="uk_b", age=36.0, comprehension=40, signed=9, spoken=12,
                 understood_only=np.nan, signed_only=3, spoken_only=6, signed_spoken=6,
                 form="DSE"),
        ]
    ).to_csv(directory / "vocab_data_uk_02.csv", index=False)
    pd.DataFrame(
        [
            dict(subject_id="uk07_a", group="control", sex="F", timepoint="t1",
                 age=40.0, understood=50, spoken=8, signed=3, spoken_signed=4,
                 produced=15, survey_vocab_max=674),
        ]
    ).to_csv(directory / "vocab_data_uk_07.csv", index=False)
    pd.DataFrame(
        [
            dict(subject_id="es_a", pair_id=1, group="DS", sex="F", age=40,
                 age_days=1200, mental_age=20.0, mental_age_level=5,
                 understood=60, spoken=20, gestured=10, spoken_or_gestured=25),
        ]
    ).to_csv(directory / "vocab_data_es_01.csv", index=False)
    pd.DataFrame(
        [
            dict(subject_id="nz_a", age=30.0, spoken=5, signed=2, spoken_signed=3),
        ]
    ).to_csv(directory / "vocab_data_nz_01.csv", index=False)


_MERGED = pd.DataFrame(
    [
        {"study": "uk_04", "age": 25.0, "understood": 30, "spoken": 20, "signed": 12,
         "subject_id": "c1", "sex": None},
        {"study": "uk_04", "age": 31.0, "understood": 38, "spoken": 24, "signed": 15,
         "subject_id": "c1", "sex": None},
        {"study": "uk_05", "age": 29.0, "understood": 41, "spoken": 27, "signed": 16,
         "subject_id": "c2", "sex": "F"},
        {"study": "ie_02", "age": 33.0, "understood": 55, "spoken": 30, "signed": 20,
         "subject_id": "c3", "sex": "M"},
    ]
)


@pytest.fixture
def build(tmp_path, monkeypatch):
    """``build(definition) -> (frame, info)`` against the fixture sources."""
    monkeypatch.setattr(env, "DATA_DIR", str(tmp_path))
    _write_sources(tmp_path)
    monkeypatch.setattr(
        cjm.vocab_data_utils,
        "load_data",
        lambda **kwargs: _MERGED[[c for c in kwargs["columns"] if c in _MERGED.columns]],
    )
    return cjm.build_joint_analysis_frame


def test_the_default_excludes_nothing_and_builds_the_same_frame(build):
    """The field's backfill entry rests on this: empty means the pre-field frame."""
    frame, info = build(VG15)
    explicit, _ = build(dataclasses.replace(VG15, exclude_studies=()))
    pd.testing.assert_frame_equal(frame, explicit)
    assert info["exclude_studies"] == ()
    assert info["excluded_study_rows"] == 0


def test_a_merged_view_study_loses_exactly_its_own_rows(build):
    frame, _ = build(VG15)
    reduced, info = build(dataclasses.replace(VG15, exclude_studies=("uk_04",)))
    assert info["excluded_study_rows"] == int((frame["study"] == "uk_04").sum()) == 2
    assert "uk_04" not in set(reduced["study"])
    assert len(reduced) == len(frame) - 2


def test_a_cross_tab_study_loses_its_cells_as_well_as_its_marginals(build):
    """The reason the filter runs on the assembled frame.

    uk_02 contributes a four-cell row to the composition and a marginal-only
    row. Filtering the merged view alone would reach neither -- uk_02 is taken
    out of the merged view and rebuilt from its own CSV -- and the study would
    stay in the fit. Both rows must go.
    """
    frame, _ = build(VG15)
    uk02 = frame["study"] == "uk_02"
    assert uk02.sum() == 2
    assert frame.loc[uk02, "signed_spoken"].notna().sum() == 1

    reduced, info = build(dataclasses.replace(VG15, exclude_studies=("uk_02",)))
    assert info["excluded_study_rows"] == 2
    assert "uk_02" not in set(reduced["study"])


def test_codes_are_assigned_after_the_rows_go(build):
    """A dropped study must not leave an empty random-effect level behind."""
    reduced, info = build(dataclasses.replace(VG15, exclude_studies=("uk_05", "es_01")))
    studies = sorted(reduced["study"].unique())
    assert info["unique_studies"] == studies
    assert sorted(reduced["study_code"].unique()) == list(range(len(studies)))
    assert sorted(reduced["subject_code"].unique()) == list(
        range(reduced["subject_code"].nunique())
    )


@pytest.mark.parametrize("excluded", ["uk_07", "ie_02", "uk_02"])
def test_study_exclusion_and_same_form_lags_preserve_likelihood_rows(
    build, tmp_path, monkeypatch, excluded
):
    """Remove a study, then restrict the remaining lags without dropping outcomes."""
    merged = _MERGED.assign(survey_vocab_max=810.0)
    monkeypatch.setattr(
        cjm.vocab_data_utils,
        "load_data",
        lambda **kwargs: merged[kwargs["columns"]].copy(),
    )
    source_path = tmp_path / "vocab_data_uk_02.csv"
    source = pd.read_csv(source_path)
    source.loc[1, ["subject_id", "form"]] = ["uk_a", "Oxford_CDI"]
    source.to_csv(source_path, index=False)

    definition = dataclasses.replace(VG25, exclude_studies=(excluded,))
    restricted = dataclasses.replace(definition, sign_lag_same_form_only=True)
    frame, _ = build(definition)
    same_form, info = build(restricted)
    pd.testing.assert_frame_equal(same_form.drop(columns="survey_vocab_max"), frame)
    assert excluded not in set(same_form["study"])
    assert sorted(same_form["study_code"].unique()) == list(
        range(len(info["unique_studies"]))
    )
    assert sorted(same_form["subject_code"].unique()) == list(
        range(info["n_subjects"])
    )

    _, original_lags, _ = prev_wave_sign_share_lag_for_frame(frame, definition)
    _, restricted_lags, _ = prev_wave_sign_share_lag_for_frame(same_form, restricted)
    dropped = (original_lags > 0) & (restricted_lags == 0)
    expected = (frame["study"] == "uk_02") & (frame["age"] == 36)
    np.testing.assert_array_equal(dropped, expected)
    assert restricted_lags.sum() == 1  # UK04's two waves still use one form.

    original_obs = prepare_joint_observations(
        frame, definition, n_trials=VG25.n_trials, use_subject_codes=True
    )
    restricted_obs = prepare_joint_observations(
        same_form, restricted, n_trials=VG25.n_trials, use_subject_codes=True
    )
    for name in ("idx_u", "idx_s", "idx_sign", "idx_cells", "idx_prod"):
        np.testing.assert_array_equal(
            getattr(original_obs, name), getattr(restricted_obs, name)
        )


def test_a_code_that_matches_nothing_is_refused(build):
    """A leave-one-study-out arm that removes nothing cannot fail."""
    with pytest.raises(ValueError, match="matched no rows"):
        build(dataclasses.replace(VG15, exclude_studies=("zz_99",)))


def test_the_prepare_stage_reports_what_was_removed(tmp_path, monkeypatch, capsys):
    """The row count is the check, so the fit's own output has to carry it."""
    import dse_research_utils.statistics.models.reporting as reporting
    import dse_research_utils.statistics.models.sampling as sampling

    from vocab_growth.models.common import ModelFitContext

    monkeypatch.setattr(env, "DATA_DIR", str(tmp_path))
    _write_sources(tmp_path)
    monkeypatch.setattr(
        cjm.vocab_data_utils,
        "load_data",
        lambda **kwargs: _MERGED[[c for c in kwargs["columns"] if c in _MERGED.columns]],
    )
    context = ModelFitContext(
        reporting=reporting.ReportingConfiguration(
            model_name="TEST_JOINT_EXCLUDE",
            config_name="test",
            output_root_dir=str(tmp_path),
            ci_prob=0.89,
            interval_kind="eti",
        ),
        sampling=sampling.get_sampling_configuration("test"),
    )
    cjm.prepare_joint_data(context, dataclasses.replace(VG15, exclude_studies=("uk_04",)))
    out = capsys.readouterr().out
    assert "Studies excluded (uk_04)" in out
    assert "2 rows" in out
