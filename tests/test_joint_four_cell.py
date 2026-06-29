# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Regression tests for the VG15 uk_02 four-cell loader.

A row that records a produced sign/speech cross-tab but is missing a cell
count — in practice ``understood_only`` (no comprehension total) — cannot form
the within-understood four-way composition and must be routed to the
marginal-only set. Otherwise the NaN cell casts to a negative integer and trips
the four-cell count validation in ``build_model``.
"""

import dse_research_utils.statistics.models.reporting as reporting
import dse_research_utils.statistics.models.sampling as sampling
import numpy as np
import pandas as pd

import vocab_growth.environment as env
from vocab_growth.models import common_joint_modality as cjm
from vocab_growth.models.common import ModelFitContext
from vocab_growth.models.definitions import VG15


def _write_uk02_csv(path):
    rows = [
        # Complete, reconciling four-cell row -> belongs in `four`.
        dict(
            subject_id="child_1", age=30.0, comprehension=19, signed=6, spoken=7,
            understood_only=10, signed_only=2, spoken_only=3, signed_spoken=4,
        ),
        # Complete and margin-reconciling, but the raw comprehension total
        # differs from the four-cell sum. This still belongs in `four`; the
        # prepared model data will use cell_total as the understood count.
        dict(
            subject_id="child_2", age=31.0, comprehension=22, signed=6, spoken=7,
            understood_only=10, signed_only=2, spoken_only=3, signed_spoken=4,
        ),
        # Reconciles on the signed/spoken margins, but understood_only (and the
        # comprehension total) is missing -> must be marginal-only.
        dict(
            subject_id="child_3", age=32.0, comprehension=np.nan, signed=6, spoken=7,
            understood_only=np.nan, signed_only=2, spoken_only=3, signed_spoken=4,
        ),
        # No cross-tab at all -> marginal-only.
        dict(
            subject_id="child_4", age=28.0, comprehension=40, signed=5, spoken=5,
            understood_only=np.nan, signed_only=np.nan, spoken_only=np.nan,
            signed_spoken=np.nan,
        ),
    ]
    pd.DataFrame(rows).to_csv(path, index=False)


def test_four_cell_loader_routes_incomplete_rows_to_marginal(tmp_path, monkeypatch):
    monkeypatch.setattr(env, "DATA_DIR", str(tmp_path))
    _write_uk02_csv(tmp_path / "vocab_data_uk_02.csv")

    four, marg = cjm._load_uk02_four_cell()

    # Only complete, margin-reconciling rows are treated as four-cell rows.
    assert len(four) == 2
    assert len(marg) == 2

    cells = ["understood_only", "signed_only", "spoken_only", "signed_spoken"]
    # All four cells present in the four-cell set...
    assert four[cells].notna().all(axis=None)
    # ...so the counts cast cleanly to non-negative integers (the original bug
    # cast a NaN cell to INT_MIN, tripping build_model's validation).
    assert np.asarray(four[cells], dtype=int).min() >= 0


def test_prepare_joint_data_uses_cell_total_and_drops_empty_rows(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(env, "DATA_DIR", str(tmp_path))
    _write_uk02_csv(tmp_path / "vocab_data_uk_02.csv")

    merged = pd.DataFrame(
        [
            {
                "study": "uk_05",
                "age": 24.0,
                "understood": np.nan,
                "spoken": np.nan,
                "signed": np.nan,
                "subject_id": "empty_child",
            },
            {
                "study": "uk_05",
                "age": 25.0,
                "understood": 30,
                "spoken": np.nan,
                "signed": np.nan,
                "subject_id": "valid_child",
            },
        ]
    )
    monkeypatch.setattr(
        cjm.vocab_data_utils,
        "load_data",
        lambda **kwargs: merged[kwargs["columns"]],
    )

    context = ModelFitContext(
        reporting=reporting.ReportingConfiguration(
            model_name="TEST_VG15_DATA",
            config_name="test",
            output_root_dir=str(tmp_path),
            hdi=0.90,
        ),
        sampling=sampling.get_sampling_configuration("test"),
    )
    cjm.prepare_joint_data(context, VG15)
    analysis_df = context.analysis_df

    assert "empty_child" not in set(analysis_df["subject_id"])
    four_rows = analysis_df[analysis_df["signed_spoken"].notna()]
    np.testing.assert_array_equal(four_rows["understood"], four_rows["cell_total"])
    assert 22 not in set(four_rows["understood"])
