# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Regression tests for the VG15 uk_02 four-cell loader.

A row that records a produced sign/speech cross-tab but is missing a cell
count — in practice ``understood_only`` (no comprehension total) — cannot form
the within-understood four-way composition and must be routed to the
marginal-only set. Otherwise the NaN cell casts to a negative integer and trips
the four-cell count validation in ``build_model``.
"""

import numpy as np
import pandas as pd

import vocab_growth.environment as env
from vocab_growth.models import common_joint_modality as cjm


def _write_uk02_csv(path):
    rows = [
        # Complete, reconciling four-cell row -> belongs in `four`.
        dict(
            age=30.0, comprehension=19, signed=6, spoken=7,
            understood_only=10, signed_only=2, spoken_only=3, signed_spoken=4,
        ),
        # Reconciles on the signed/spoken margins, but understood_only (and the
        # comprehension total) is missing -> must be marginal-only.
        dict(
            age=32.0, comprehension=np.nan, signed=6, spoken=7,
            understood_only=np.nan, signed_only=2, spoken_only=3, signed_spoken=4,
        ),
        # No cross-tab at all -> marginal-only.
        dict(
            age=28.0, comprehension=40, signed=5, spoken=5,
            understood_only=np.nan, signed_only=np.nan, spoken_only=np.nan,
            signed_spoken=np.nan,
        ),
    ]
    pd.DataFrame(rows).to_csv(path, index=False)


def test_four_cell_loader_routes_incomplete_rows_to_marginal(tmp_path, monkeypatch):
    monkeypatch.setattr(env, "DATA_DIR", str(tmp_path))
    _write_uk02_csv(tmp_path / "vocab_data_uk_02.csv")

    four, marg = cjm._load_uk02_four_cell()

    # Only the complete, reconciling row is treated as a four-cell row.
    assert len(four) == 1
    assert len(marg) == 2

    cells = ["understood_only", "signed_only", "spoken_only", "signed_spoken"]
    # All four cells present in the four-cell set...
    assert four[cells].notna().all(axis=None)
    # ...so the counts cast cleanly to non-negative integers (the original bug
    # cast a NaN cell to INT_MIN, tripping build_model's validation).
    assert np.asarray(four[cells], dtype=int).min() >= 0
