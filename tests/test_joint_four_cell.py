# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Regression tests for the VG15 uk_02 four-cell loader.

A row that records a produced sign/speech cross-tab but is missing a cell
count — in practice ``understood_only`` (no comprehension total) — cannot form
the within-understood four-way composition and must be routed to the
marginal-only set. Otherwise the NaN cell casts to a negative integer and trips
the four-cell count validation in ``build_model``.
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


def _write_uk07_csv(path, rows=None):
    """A uk_07 fixture — prepare_joint_data always loads this CSV."""
    if rows is None:
        rows = [
            dict(
                subject_id="uk07_c", group="control", sex="F", timepoint="t1",
                age=40.0, understood=50, spoken=8, signed=3, spoken_signed=4,
                produced=15, survey_vocab_max=674,
            )
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
    _write_uk07_csv(tmp_path / "vocab_data_uk_07.csv")

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
            ci_prob=0.90,
            interval_kind="hdi",
        ),
        sampling=sampling.get_sampling_configuration("test"),
    )
    cjm.prepare_joint_data(context, VG15)
    analysis_df = context.analysis_df

    assert "empty_child" not in set(analysis_df["subject_id"])
    four_rows = analysis_df[analysis_df["signed_spoken"].notna()]
    np.testing.assert_array_equal(four_rows["understood"], four_rows["cell_total"])
    assert 22 not in set(four_rows["understood"])


def test_uk07_loader_derives_the_fourth_cell_and_guards_the_partition(
    tmp_path, monkeypatch
):
    """uk_07's fourth cell is ``understood - produced``; a row where that would be
    negative (production above comprehension) or where nothing is understood
    carries no within-understood composition and must fall back to marginals."""
    monkeypatch.setattr(env, "DATA_DIR", str(tmp_path))
    _write_uk07_csv(
        tmp_path / "vocab_data_uk_07.csv",
        rows=[
            # 50 understood, 15 produced -> understood_only = 35. Four-cell.
            dict(
                subject_id="ok", group="control", sex="F", timepoint="t1",
                age=40.0, understood=50, spoken=8, signed=3, spoken_signed=4,
                produced=15, survey_vocab_max=674,
            ),
            # Production above comprehension -> no non-negative fourth cell.
            dict(
                subject_id="over", group="control", sex="M", timepoint="t3",
                age=58.0, understood=10, spoken=8, signed=3, spoken_signed=4,
                produced=15, survey_vocab_max=674,
            ),
            # Nothing understood -> no composition at all.
            dict(
                subject_id="zero", group="intervention", sex="M", timepoint="t1",
                age=36.0, understood=0, spoken=0, signed=0, spoken_signed=0,
                produced=0, survey_vocab_max=674,
            ),
        ],
    )

    four, marg = cjm._load_uk07_four_cell()

    assert list(four["subject_id"]) == ["ok"]
    assert sorted(marg["subject_id"]) == ["over", "zero"]
    row = four.iloc[0]
    assert row["understood_only"] == 35
    # The four cells partition the understood total exactly, so cell_total is the
    # recorded comprehension count rather than something to reconcile against it.
    assert (
        row["understood_only"] + row["signed"] + row["spoken"] + row["spoken_signed"]
        == row["understood"]
    )


def test_uk07_cells_join_uk02_in_the_psi_likelihood(tmp_path, monkeypatch):
    """With ``include_uk07_cells`` on, uk_07 rows enter as four-cell rows and their
    marginals are dropped from the merged view (no double counting); with it off,
    uk_07 stays in the fit through its marginals — unlike nz_01, which leaves."""
    monkeypatch.setattr(env, "DATA_DIR", str(tmp_path))
    _write_uk02_csv(tmp_path / "vocab_data_uk_02.csv")
    _write_uk07_csv(tmp_path / "vocab_data_uk_07.csv")

    merged = pd.DataFrame(
        [
            {
                "study": "uk_07",
                "age": 40.0,
                "understood": 50,
                "spoken": 12,
                "signed": 7,
                "subject_id": "uk07_c",
            },
            {
                "study": "uk_05",
                "age": 25.0,
                "understood": 30,
                "spoken": 5,
                "signed": 2,
                "subject_id": "valid_child",
            },
        ]
    )
    monkeypatch.setattr(
        cjm.vocab_data_utils,
        "load_data",
        lambda **kwargs: merged[kwargs["columns"]],
    )

    def _prepared(definition):
        context = ModelFitContext(
            reporting=reporting.ReportingConfiguration(
                model_name="TEST_VG15_UK07",
                config_name="test",
                output_root_dir=str(tmp_path),
                ci_prob=0.90,
                interval_kind="hdi",
            ),
            sampling=sampling.get_sampling_configuration("test"),
        )
        cjm.prepare_joint_data(context, definition)
        return context.analysis_df

    on = _prepared(VG15)
    uk07_on = on[on["study"] == "uk_07"]
    assert len(uk07_on) == 1
    assert uk07_on.iloc[0]["signed_spoken"] == 4          # a four-cell row
    assert pd.isna(uk07_on.iloc[0]["spoken"])             # marginals suppressed
    assert pd.isna(uk07_on.iloc[0]["signed"])
    # It is a within-understood cross-tab, so it joins uk_02 in the same term.
    cell_studies = set(on.loc[on["signed_spoken"].notna(), "study"])
    assert cell_studies == {"uk_02", "uk_07"}

    off = _prepared(dataclasses.replace(VG15, include_uk07_cells=False))
    uk07_off = off[off["study"] == "uk_07"]
    assert len(uk07_off) == 1
    assert pd.isna(uk07_off.iloc[0]["signed_spoken"])     # no cross-tab term
    assert uk07_off.iloc[0]["spoken"] == 12               # marginals retained
    assert uk07_off.iloc[0]["signed"] == 7


def test_plackett_pi_both_stable_and_correct_at_psi_one():
    """The rationalised Plackett root is continuous and correctly-differentiable
    at psi == 1 (issue #131 §3): it returns the independence value ``r*q`` with a
    finite, non-zero gradient, where the old ``switch`` form returned a spurious
    zero/NaN gradient in the psi->1 neighbourhood."""
    import pytensor
    import pytensor.tensor as pt

    psi = pt.dscalar("psi")
    r, q = 0.4, 0.6  # r + q - 1 = 0, so the Frechet lower bound is 0 here
    expr = cjm._plackett_pi_both(r, q, psi)
    f = pytensor.function([psi], [expr, pt.grad(expr, psi)])

    val, grad = f(1.0)
    assert np.isclose(val, r * q)  # independence at psi == 1
    assert np.isfinite(grad) and abs(grad) > 1e-6  # not the old spurious 0

    # Matches the textbook closed-form root away from psi == 1.
    for p in (0.5, 2.0, 5.0):
        S = 1.0 + (r + q) * (p - 1.0)
        disc = np.sqrt(S * S - 4.0 * p * (p - 1.0) * r * q)
        textbook = (S - disc) / (2.0 * (p - 1.0))
        assert np.isclose(f(p)[0], textbook)
