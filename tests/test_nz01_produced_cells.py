# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Regression tests for the VG15 nz_01 (Foster-Cohen) produced three-cell path.

nz_01 is production-only (no comprehension): its checklist partitions produced
items into word-only, sign-only and both. Conditioned on production those three
cells form a within-produced Dirichlet-Multinomial that informs ``psi`` (see
``common_joint_modality.build_model``). These tests pin the data-prep path — the
loader's column mapping and zero-produced drop, and ``prepare_joint_data``'s
inclusion behind the ``include_nz01_cells`` gate without double-counting the
production-only marginal.
"""

import dataclasses

import dse_research_utils.statistics.models.reporting as reporting
import dse_research_utils.statistics.models.sampling as sampling
import numpy as np
import pandas as pd

import vocab_growth.environment as env
from vocab_growth.models import common_joint_modality as cjm
from vocab_growth.models.common import ModelFitContext
from vocab_growth.models.definitions import VG15


def _write_nz01_csv(path):
    rows = [
        # word-only=5, sign-only=2, both=3 -> produced 10
        dict(subject_id="nz_1", age=30.0, spoken=5, signed=2, spoken_signed=3),
        # word-only=8, sign-only=0, both=1 -> produced 9
        dict(subject_id="nz_2", age=48.0, spoken=8, signed=0, spoken_signed=1),
        # no produced words at all -> must be dropped (carries no composition)
        dict(subject_id="nz_3", age=40.0, spoken=0, signed=0, spoken_signed=0),
    ]
    pd.DataFrame(rows).to_csv(path, index=False)


def test_nz01_loader_maps_cells_and_drops_zero_produced(tmp_path, monkeypatch):
    monkeypatch.setattr(env, "DATA_DIR", str(tmp_path))
    _write_nz01_csv(tmp_path / "vocab_data_nz_01.csv")

    out = cjm._load_nz01_produced_cells()

    # The zero-produced row is dropped; two rows remain.
    assert set(out["subject_id"]) == {"nz_1", "nz_2"}
    assert (out["study"] == cjm.NZ01_STUDY_ID).all()
    # Modality-exclusive CSV columns map straight through (spoken=word-only,
    # signed=sign-only, spoken_signed=both).
    r1 = out[out["subject_id"] == "nz_1"].iloc[0]
    assert (
        r1["prod_spoken_only"],
        r1["prod_signed_only"],
        r1["prod_signed_spoken"],
    ) == (5, 2, 3)
    # prod_total is the sum of the three produced cells.
    np.testing.assert_array_equal(
        out["prod_total"],
        out["prod_spoken_only"] + out["prod_signed_only"] + out["prod_signed_spoken"],
    )


def _prepare(tmp_path, monkeypatch, definition):
    """Run ``prepare_joint_data`` for ``definition`` with nz_01 + uk_02 fixtures."""
    monkeypatch.setattr(env, "DATA_DIR", str(tmp_path))
    _write_nz01_csv(tmp_path / "vocab_data_nz_01.csv")
    # prepare_joint_data always loads the uk_02 four-cell CSV.
    pd.DataFrame(
        [
            dict(
                subject_id="uk_c", age=30.0, comprehension=19, signed=6, spoken=7,
                understood_only=10, signed_only=2, spoken_only=3, signed_spoken=4,
            )
        ]
    ).to_csv(tmp_path / "vocab_data_uk_02.csv", index=False)

    # A few complete marginal DS rows so the data-prep summary's per-column
    # normality checks have observations on every outcome (an all-NaN column
    # trips a division-by-zero in the shared Anderson-Darling helper) -> kept in
    # `other`.
    merged = pd.DataFrame(
        [
            {"study": "uk_04", "age": 25.0, "understood": 30, "spoken": 20,
             "signed": 12, "subject_id": "child_a"},
            {"study": "uk_04", "age": 27.0, "understood": 35, "spoken": 23,
             "signed": 14, "subject_id": "child_b"},
            {"study": "uk_05", "age": 29.0, "understood": 41, "spoken": 27,
             "signed": 16, "subject_id": "child_c"},
            # a production-only nz_01 MARGINAL row -> must be excluded from `other`
            # so it is not double-counted against the produced-cell DM
            {"study": cjm.NZ01_STUDY_ID, "age": 33.0, "understood": np.nan,
             "spoken": 12, "signed": 4, "subject_id": "nz_marginal_only"},
        ]
    )
    monkeypatch.setattr(
        cjm.vocab_data_utils, "load_data", lambda **kwargs: merged[kwargs["columns"]]
    )
    context = ModelFitContext(
        reporting=reporting.ReportingConfiguration(
            model_name="TEST_VG15_NZ01",
            config_name="test",
            output_root_dir=str(tmp_path),
            hdi=0.90,
        ),
        sampling=sampling.get_sampling_configuration("test"),
    )
    cjm.prepare_joint_data(context, definition)
    return context.analysis_df


def test_prepare_joint_data_includes_nz01_produced_cells(tmp_path, monkeypatch):
    analysis_df = _prepare(tmp_path, monkeypatch, VG15)

    # The two non-zero-produced nz_01 rows enter as produced-cell rows.
    prod_rows = analysis_df[analysis_df["prod_signed_spoken"].notna()]
    assert len(prod_rows) == 2
    # The production-only nz_01 marginal is not double-counted (excluded from `other`).
    assert "nz_marginal_only" not in set(analysis_df["subject_id"])
    # Produced cells still sum to prod_total after prep.
    np.testing.assert_array_equal(
        prod_rows["prod_total"],
        prod_rows["prod_spoken_only"]
        + prod_rows["prod_signed_only"]
        + prod_rows["prod_signed_spoken"],
    )


def test_prepare_joint_data_excludes_nz01_when_flag_false(tmp_path, monkeypatch):
    definition = dataclasses.replace(VG15, include_nz01_cells=False)
    analysis_df = _prepare(tmp_path, monkeypatch, definition)

    # With the gate off, no produced-cell rows are added...
    if "prod_signed_spoken" in analysis_df.columns:
        assert analysis_df["prod_signed_spoken"].notna().sum() == 0
    # ...and the nz_01 marginal is still excluded (not silently re-added as marginal).
    assert "nz_marginal_only" not in set(analysis_df["subject_id"])
