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


def _write_es01_csv(path, rows=None):
    """An es_01 fixture — prepare_joint_data always loads this CSV."""
    if rows is None:
        rows = [
            dict(
                subject_id="es_c", pair_id=1, group="DS", sex="F", age=40,
                age_days=1200, mental_age=20.0, mental_age_level=5,
                understood=60, spoken=20, gestured=10, spoken_or_gestured=25,
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
    _write_es01_csv(tmp_path / "vocab_data_es_01.csv")

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
    _write_es01_csv(tmp_path / "vocab_data_es_01.csv")
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
    _write_es01_csv(tmp_path / "vocab_data_es_01.csv")

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
    assert cell_studies == {"uk_02", "uk_07", "es_01"}

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


# ---- es_01 (Galeote): four cells derived from the recorded union ----
#
# The source's fourth column is a recorded UNION ("WORD PRODUCED + GESTURES ONLY"
# in the original table; "total lexical production combining the two modalities"
# in Galeote et al. 2011), and its third is a gestural TOTAL. So the four cells
# follow by subtraction and sum to `understood` identically. A disjoint reading of
# the third column would force union == spoken + gestured, which 134 of the 186
# real rows violate.


def test_es01_loader_derives_four_cells_from_the_recorded_union(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(env, "DATA_DIR", str(tmp_path))
    _write_es01_csv(
        tmp_path / "vocab_data_es_01.csv",
        rows=[
            # understood 60, spoken 20, gestured 10, union 25
            #   -> both = 20 + 10 - 25 = 5, spoken_only = 25 - 10 = 15,
            #      gesture_only = 25 - 20 = 5, understood_only = 60 - 25 = 35
            dict(
                subject_id="ok", pair_id=1, group="DS", sex="F", age=40,
                age_days=1200, mental_age=20.0, mental_age_level=5,
                understood=60, spoken=20, gestured=10, spoken_or_gestured=25,
            ),
            # The real defective row: a union smaller than one of its parts, so
            # spoken_only = 11 - 15 is negative -> marginal-only.
            dict(
                subject_id="bad", pair_id=148, group="DS", sex="M", age=24,
                age_days=720, mental_age=12.7, mental_age_level=2,
                understood=82, spoken=1, gestured=15, spoken_or_gestured=11,
            ),
            # The matched typically developing partner must never reach this
            # relation, whatever its counts.
            dict(
                subject_id="td", pair_id=1, group="TD", sex="F", age=20,
                age_days=600, mental_age=20.0, mental_age_level=5,
                understood=60, spoken=20, gestured=10, spoken_or_gestured=25,
            ),
        ],
    )

    four, marg = cjm._load_es01_four_cell()

    assert list(four["subject_id"]) == ["ok"]
    assert list(marg["subject_id"]) == ["bad"]          # TD is filtered, not routed

    row = four.iloc[0]
    assert row["understood_only"] == 35
    assert row["spoken_only"] == 15
    assert row["signed_only"] == 5
    assert row["signed_spoken"] == 5
    # The defining property: the partition is exhaustive within understood.
    assert (
        row["understood_only"] + row["spoken_only"]
        + row["signed_only"] + row["signed_spoken"]
    ) == row["understood"]


def test_es01_cells_join_the_psi_likelihood_and_fall_back_when_off(
    tmp_path, monkeypatch
):
    """With ``include_es01_cells`` on, es_01 enters as four-cell rows with its
    marginals suppressed; with it off it stays in the fit through its marginals."""
    monkeypatch.setattr(env, "DATA_DIR", str(tmp_path))
    _write_uk02_csv(tmp_path / "vocab_data_uk_02.csv")
    _write_uk07_csv(tmp_path / "vocab_data_uk_07.csv")
    _write_es01_csv(tmp_path / "vocab_data_es_01.csv")

    merged = pd.DataFrame(
        [
            {"study": "es_01", "age": 40.0, "understood": 60, "spoken": 20,
             "signed": 10, "subject_id": "es_c"},
            {"study": "uk_05", "age": 25.0, "understood": 30, "spoken": 5,
             "signed": 2, "subject_id": "valid_child"},
        ]
    )
    monkeypatch.setattr(
        cjm.vocab_data_utils, "load_data", lambda **kwargs: merged[kwargs["columns"]]
    )

    def _prepared(definition):
        context = ModelFitContext(
            reporting=reporting.ReportingConfiguration(
                model_name="TEST_VG15_ES01",
                config_name="test",
                output_root_dir=str(tmp_path),
                ci_prob=0.90,
                interval_kind="hdi",
            ),
            sampling=sampling.get_sampling_configuration("test"),
        )
        cjm.prepare_joint_data(context, definition)
        return context.analysis_df

    on = _prepared(VG15)                       # default: cells on
    es_on = on[on["study"] == "es_01"]
    assert len(es_on) == 1
    assert es_on.iloc[0]["signed_spoken"] == 5           # a four-cell row
    assert pd.isna(es_on.iloc[0]["spoken"])              # marginals suppressed
    assert pd.isna(es_on.iloc[0]["signed"])

    off = _prepared(dataclasses.replace(VG15, include_es01_cells=False))
    es_off = off[off["study"] == "es_01"]
    assert len(es_off) == 1
    assert pd.isna(es_off.iloc[0]["signed_spoken"])      # no cross-tab term
    assert es_off.iloc[0]["spoken"] == 20                # marginals retained
    assert es_off.iloc[0]["signed"] == 10


def test_es01_defective_row_keeps_its_marginals_but_not_its_gestural_total(
    tmp_path, monkeypatch
):
    """A row lands in the marginal set only because its cells do not reconcile —
    which is exactly the condition under which the view masks its gestural total.
    Passing that total through here would reintroduce what the view rejects."""
    monkeypatch.setattr(env, "DATA_DIR", str(tmp_path))
    _write_uk02_csv(tmp_path / "vocab_data_uk_02.csv")
    _write_uk07_csv(tmp_path / "vocab_data_uk_07.csv")
    _write_es01_csv(
        tmp_path / "vocab_data_es_01.csv",
        rows=[
            dict(
                subject_id="bad", pair_id=148, group="DS", sex="M", age=24,
                age_days=720, mental_age=12.7, mental_age_level=2,
                understood=82, spoken=1, gestured=15, spoken_or_gestured=11,
            )
        ],
    )
    monkeypatch.setattr(
        cjm.vocab_data_utils,
        "load_data",
        lambda **kwargs: pd.DataFrame(
            [{"study": "uk_05", "age": 25.0, "understood": 30, "spoken": 5,
              "signed": 2, "subject_id": "valid_child"}]
        )[kwargs["columns"]],
    )

    context = ModelFitContext(
        reporting=reporting.ReportingConfiguration(
            model_name="TEST_VG15_ES01_BAD",
            config_name="test",
            output_root_dir=str(tmp_path),
            ci_prob=0.90,
            interval_kind="hdi",
        ),
        sampling=sampling.get_sampling_configuration("test"),
    )
    cjm.prepare_joint_data(context, VG15)
    row = context.analysis_df.query("study == 'es_01'").iloc[0]

    assert pd.isna(row["signed_spoken"])   # no composition
    assert row["understood"] == 82         # comprehension survives
    assert row["spoken"] == 1              # oral production survives
    assert pd.isna(row["signed"])          # the impossible gestural total does not


def test_psi_carries_a_study_term_and_all_cross_tab_sources_are_on():
    """psi was once the only latent here with no study-level term.

    delta_u, delta_q and delta_sign were all study random intercepts while log_psi
    was a bare global scalar — so the reported association was a precision-weighted
    average over whichever cross-tab sources were in the pool, and it moved 1.80 to
    2.49 on adding uk_07 alone. The sources disagree far more than that
    (Mantel-Haenszel, stratified by child: uk_02 6.09, uk_07 13.90, nz_01 14.63,
    es_01 0.90).

    With delta_psi in place the heterogeneity is estimated rather than averaged
    away, so every cross-tab source is admitted. This pins both halves: turning a
    source back off must stay possible, but the defaults must not silently revert
    to pooling into a study-invariant psi.
    """
    assert VG15.include_uk07_cells is True
    assert VG15.include_es01_cells is True
    assert VG15.include_nz01_cells is True
    # The between-study scale is wider than the trajectory taus (0.5) because the
    # measured spread is wider; a tighter prior would fight the data.
    assert VG15.tau_psi_sigma == 1.0


def test_es01_real_cells_reconcile_and_sit_near_independence():
    """The real source, not a fixture: the partition must be exact, and the
    association must still be the one the default is justified on."""
    four, marg = cjm._load_es01_four_cell()

    assert len(four) == 185 and len(marg) == 1     # one known defective row

    cells = ["understood_only", "spoken_only", "signed_only", "signed_spoken"]
    # Exhaustive within understood, on every row, with no negative cell.
    assert (four[cells].sum(axis=1) == four["understood"]).all()
    assert (four[cells] >= 0).all().all()

    # Mantel-Haenszel across children, the estimator quoted in the flag docstring.
    total = four[cells].sum(axis=1)
    num = (four["signed_spoken"] * four["understood_only"] / total).sum()
    den = (four["signed_only"] * four["spoken_only"] / total).sum()
    assert 0.7 < num / den < 1.2, (
        "es_01's sign-speech association has moved away from independence; "
        "the include_es01_cells default was justified on it sitting there while "
        "the other cross-tab sources sit well above it."
    )


def test_dse_native_only_restricts_the_pool_and_collapses_psi_to_uk02():
    """The DSE-native sensitivity on the real sources, through the real engine.

    Three things have to hold together, and only the third is obvious from the
    flag. The merged view must lose every source on a shorter form. The three
    cross-tab blocks that read their own CSVs must be gated off too — a row filter
    on the merged view would never see them, so without the gate uk_07, es_01 and
    nz_01 would slip back in through the side door carrying exactly the
    harmonisation the variant exists to remove. And uk_02, which ran both
    instruments, must keep its DSE arm alone.
    """
    context = ModelFitContext(
        reporting=reporting.ReportingConfiguration(
            model_name="TEST_VG15_NATIVE",
            config_name="test",
            output_root_dir=str(env.OUTPUT_DIR),
            ci_prob=0.90,
            interval_kind="hdi",
        ),
        sampling=sampling.get_sampling_configuration("test"),
    )
    native = dataclasses.replace(VG15, dse_native_only=True)
    cjm.prepare_joint_data(context, native)
    df = context.analysis_df

    # Only the four sources whose form IS the 810-item reference.
    assert set(df["study"]) == {"ie_01", "ie_02", "uk_02", "uk_06"}

    # The cross-tab side door: uk_07 and es_01 are on 674- and 651-item forms, so
    # their cells must be absent even though both inclusion flags are still True.
    assert native.include_uk07_cells and native.include_es01_cells
    cells = df[df["signed_spoken"].notna()]
    assert set(cells["study"]) == {"uk_02"}
    # All 56 of uk_02's four-cell rows are its DSE arm, so the cross-tab survives
    # whole and only Oxford marginals leave.
    assert len(cells) == 56
    # nz_01's within-produced cells go the same way: its 675-item NZCDI is not the
    # reference form, so its three-cell block contributes nothing here either.
    assert "nz_01" not in set(df["study"])
    if "prod_total" in df.columns:
        assert df["prod_total"].notna().sum() == 0

    # psi therefore has one informed study and falls back to its single-study
    # branch. The variant answers the denominator question and the between-study
    # question at once and cannot separate them — which is why it is documented as
    # a trajectory-shape check rather than a psi check.
    assert cells["study"].nunique() == 1


# ------------------------------------------------- sign-to-speech milestones


def test_signing_milestones_recover_a_known_hand_over():
    """A synthetic hand-over with a known peak is recovered per draw.

    The peak is deliberately jittered across draws: reading it off the median
    curve instead would flatten and re-centre it, which is the error this
    function exists to avoid.
    """
    import numpy as np

    from vocab_growth.models.common_joint_modality import _signing_milestones

    ages = np.arange(8.0, 84.0, 0.25)
    rng = np.random.default_rng(11)
    n_draw = 300
    peaks = 34.0 + rng.normal(0.0, 1.5, n_draw)
    sign_only = 50.0 * np.exp(-((ages[:, None] - peaks[None, :]) ** 2) / (2 * 10.0**2))
    speak_only = 200.0 / (1.0 + np.exp(-(ages[:, None] - 50.0) / 6.0))
    speak_only = np.repeat(speak_only, 1, axis=1) * np.ones((1, n_draw))
    both = np.full_like(sign_only, 5.0)

    got = _signing_milestones(ages, sign_only, both, speak_only, 0.89)
    got = got.set_index("quantity")

    assert abs(got.loc["sign_only_peak_age", "median"] - 34.0) < 1.0
    assert got.loc["sign_only_peak_age", "ci_lo"] < 34.0 < got.loc["sign_only_peak_age", "ci_hi"]
    assert abs(got.loc["sign_only_peak_words", "median"] - 50.0) < 1.0
    # Ordering: the peak comes first, then the crossings.
    assert (
        got.loc["sign_only_peak_age", "median"]
        < got.loc["speech_only_overtakes_sign_only_age", "median"]
    )
    assert (got["draws_reaching"] == 1.0).all()
    assert (got["draws_censored"] == 0.0).all()


def test_signing_milestones_classify_always_true_states_as_censored():
    """A state already true at establishment is censored, not an early crossing.

    Below a word of expressive vocabulary, cell ordering is arithmetic noise, so
    milestones are only read once the draw's child has a vocabulary to divide up
    (a word-count gate, not a grid-point one, so it cannot depend on the step
    size). But the gate alone is not enough: in this scenario speech leads sign
    at every established age, so there is no overtake to date — the pre-#238
    rule nevertheless reported the first established age as one.
    """
    import numpy as np

    from vocab_growth.models.common_joint_modality import _signing_milestones

    ages = np.arange(8.0, 84.0, 0.25)
    n_draw = 20
    # Speech exceeds sign everywhere, but both are negligible until ~40 months.
    ramp = np.clip((ages - 40.0) / 40.0, 0.0, None)[:, None] * np.ones((1, n_draw))
    sign_only = 10.0 * ramp
    speak_only = 20.0 * ramp
    both = np.zeros_like(sign_only)

    got = _signing_milestones(ages, sign_only, both, speak_only, 0.89).set_index("quantity")
    row = got.loc["speech_only_overtakes_sign_only_age"]
    # Speech leads sign from the first established age in every draw, so there
    # is no overtake TRANSITION anywhere on the grid: the state is left-censored,
    # not an early crossing. The pre-#238 rule reported the first established
    # age (~41 months) as an overtake, which this scenario never contains.
    assert row["draws_reaching"] == 0.0
    assert row["draws_censored"] == 1.0
    assert np.isnan(row["median"])


def test_signing_milestones_flag_a_milestone_never_reached():
    """draws_reaching is the identification warning, not decoration."""
    import numpy as np

    from vocab_growth.models.common_joint_modality import _signing_milestones

    ages = np.arange(8.0, 84.0, 0.25)
    n_draw = 10
    sign_only = np.full((len(ages), n_draw), 60.0)
    speak_only = np.full((len(ages), n_draw), 5.0)  # never overtakes
    both = np.zeros_like(sign_only)

    got = _signing_milestones(ages, sign_only, both, speak_only, 0.89).set_index("quantity")
    assert got.loc["speech_only_overtakes_sign_only_age", "draws_reaching"] == 0.0
    # Never true is distinct from censored: the condition never held, so the
    # state was not already-true at establishment either.
    assert got.loc["speech_only_overtakes_sign_only_age", "draws_censored"] == 0.0
    assert np.isnan(got.loc["speech_only_overtakes_sign_only_age", "median"])
