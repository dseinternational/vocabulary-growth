# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""VG25's sign -> speech cross-lag on the joint engine (issue #297).

VG25 is VG24 with one added coefficient: a child's prior-wave **signed share of
comprehension**, relative to their own persistent signing standing, shifts the
logit of their current production ratio ``q``. VG24 is nested exactly at
``beta_sign_lag = 0``.

Four properties carry the design and are pinned here.

- **The coefficient actually reaches the density.** VG24's first build reported
  all three of its correlations as ``+0.000`` in every draw because the
  primitive returned the wrong object, and a model whose headline is silently
  pinned at zero is indistinguishable from one that fitted and found nothing
  (``notes/202609061530``). ``beta_sign_lag`` is the same shape of parameter, so
  :func:`test_the_coefficient_reaches_every_likelihood_it_claims_to` measures the
  logp's dependence on it rather than reading the source.
- **VG24 is nested exactly at zero**, which is what makes "did anything else
  move?" a meaningful question of the comparison.
- **Turning the lag off leaves the graph VG24's, op for op**, so registering
  VG25 cannot have moved any other model -- the property the graph baseline
  records and this checks directly.
- **The predictor reads what it claims to.** The signed share has two sources in
  the joint frame (the marginal and the within-understood cells) and one
  near-miss that must not be a third (``nz_01``'s produced cells, which
  partition a different denominator).

The definition-subclass check matters as much and is cheap: putting these fields
on ``JointCorrelatedSubjectREModelDefinition`` would change VG24's serialised
definition and invalidate every VG24 fit on disk.
"""

import os
import types
from dataclasses import fields

import dse_research_utils.statistics.models.data as model_data
import dse_research_utils.statistics.models.pymc_utils as pymc_utils
import dse_research_utils.statistics.models.reporting as reporting
import dse_research_utils.statistics.models.sampling as sampling
import numpy as np
import pandas as pd
import pytest
from support.synthetic_graphs import synthetic_frame

from vocab_growth.models.catalogue import ENGINES
from vocab_growth.models.common import ModelFitContext
from vocab_growth.models.cross_lag import (
    prev_wave_lag,
    prev_wave_sign_share_lag,
    prev_wave_sign_share_lag_for_frame,
    sign_share_counts,
    validate_sign_cross_lag,
)
from vocab_growth.models.definitions import (
    VG24,
    VG25,
    JointCorrelatedSubjectREModelDefinition,
    JointCrossLagModelDefinition,
    _as_definition_subclass,
    validate_model_definition,
)
from vocab_growth.models.likelihood_utils import (
    LAG_ZERO_CLIP,
    LAG_ZERO_CONTINUITY,
)

LOGIT_CLIP_HIGH = np.log(1 - 1e-4) - np.log(1e-4)

#: Everything above this line is arithmetic on arrays and costs nothing. The
#: graph section at the bottom builds real PyMC models, is marked `slow`, and
#: carries an `xdist_group` for the reason `test_graph_equivalence` records: its
#: tests share module-scoped built graphs, so a per-test distribution would
#: rebuild them once per worker that happens to draw one.
GRAPH_TESTS = [pytest.mark.slow, pytest.mark.xdist_group("joint-sign-cross-lag")]


_JOINT_ENGINE = ENGINES["joint"]


def _logit(value):
    return float(np.log(value) - np.log1p(-value))


# ============================================================
# The definition
# ============================================================


def test_vg25_differs_from_vg24_only_in_naming_and_the_lag():
    """The two models must differ in the lag fields and nothing else."""
    v24 = {f.name: getattr(VG24, f.name) for f in fields(VG24)}
    v25 = {f.name: getattr(VG25, f.name) for f in fields(VG25)}
    changed = {
        k
        for k in set(v24) | set(v25)
        if v24.get(k, "<absent>") != v25.get(k, "<absent>")
    }
    assert changed == {
        "model_id",
        "config_name",
        "banner",
        "use_sign_cross_lag",
        "sign_lag_baseline",
        "sign_lag_in_cells",
        "beta_sign_lag_mu",
        "beta_sign_lag_sigma",
        "sign_lag_max_gap_months",
        "sign_lag_zero_handling",
    }


def test_vg24_does_not_gain_the_fields():
    """The subclass must not leak onto the parent class.

    A fit is validated by comparing the serialised definition field for field,
    so if this ever fails every fitted VG24 output on disk becomes invalid at the
    same moment -- and VG24 is a model of record.
    """
    parent_fields = {f.name for f in fields(VG24)}
    for name in (
        "use_sign_cross_lag",
        "sign_lag_baseline",
        "sign_lag_in_cells",
        "beta_sign_lag_mu",
        "beta_sign_lag_sigma",
        "sign_lag_max_gap_months",
        "sign_lag_zero_handling",
    ):
        assert name not in parent_fields, name
    assert type(VG24) is JointCorrelatedSubjectREModelDefinition


def test_vg25_is_the_subclass_and_inherits_the_correlated_block():
    """Deriving from VG24 rather than VG15 is the whole interpretive argument.

    `rho_sign_q` is what takes the persistent sign-speech association off the
    lag; without it in the same model the coefficient is a noisy proxy for it
    (`notes/202608151140` s3). So this is a statistical claim, not a class
    hierarchy detail.
    """
    assert isinstance(VG25, JointCrossLagModelDefinition)
    assert isinstance(VG25, JointCorrelatedSubjectREModelDefinition)
    assert VG25.subject_re_correlation_eta == VG24.subject_re_correlation_eta
    assert VG25.use_sign_cross_lag


def test_the_registered_choices_are_the_ones_the_record_states():
    """Each of these was a decision with a recorded reason; pin them together.

    A silent flip of any one changes the estimand or the evidence: the baseline
    decides whether the coefficient duplicates `rho_sign_q`, the cells decide
    whether uk_07 contributes at all, and the zero treatment decides whether
    76% of the predictor's sum of squares comes from a floor constant.
    """
    assert VG25.sign_lag_baseline == "within"
    assert VG25.sign_lag_in_cells is True
    assert VG25.sign_lag_zero_handling == LAG_ZERO_CONTINUITY
    assert VG25.sign_lag_max_gap_months is None


def test_the_lag_prior_matches_vg16s():
    """Deliberately the same Normal(0, 0.5), so the two coefficients compare."""
    from vocab_growth.models.definitions import VG16

    assert (VG25.beta_sign_lag_mu, VG25.beta_sign_lag_sigma) == (
        VG16.beta_lag_mu,
        VG16.beta_lag_sigma,
    )


def test_a_sign_lag_without_a_sign_child_effect_is_rejected():
    """Both baselines are defined relative to the signed-ratio intercept.

    Without one they silently coincide, which would make the registered
    within-child model and its population-relative sensitivity the same fit
    under two names.
    """
    definition = _as_definition_subclass(
        VG25,
        JointCrossLagModelDefinition,
        model_id="VG99",
        use_subject_re_sign=False,
    )
    with pytest.raises(ValueError, match="use_subject_re_sign=True"):
        validate_model_definition(definition)


@pytest.mark.parametrize("bad", ["", "population-relative", "Within", None])
def test_an_unknown_sign_lag_baseline_is_rejected(bad):
    definition = _as_definition_subclass(
        VG25, JointCrossLagModelDefinition, model_id="VG99", sign_lag_baseline=bad
    )
    with pytest.raises(ValueError, match="sign_lag_baseline"):
        validate_model_definition(definition)


@pytest.mark.parametrize("bad", ["", "clamp", "haldane"])
def test_an_unknown_sign_lag_zero_handling_is_rejected(bad):
    """Rejected at definition time, not inside the array primitive.

    The primitive is reached after data preparation and prior configuration have
    already run, which is the defect `lag_zero_handling` had until 2026-09-01.
    """
    definition = _as_definition_subclass(
        VG25, JointCrossLagModelDefinition, model_id="VG99", sign_lag_zero_handling=bad
    )
    with pytest.raises(ValueError, match="sign_lag_zero_handling"):
        validate_model_definition(definition)


def test_the_engine_side_check_names_its_own_flag():
    """Two lags, two messages: one naming the wrong flag is worse than none."""
    validate_sign_cross_lag("within", subject_re_sign_active=True)
    validate_sign_cross_lag("population", subject_re_sign_active=True)
    with pytest.raises(ValueError, match="use_sign_cross_lag"):
        validate_sign_cross_lag("within", subject_re_sign_active=False)
    with pytest.raises(ValueError, match="sign_lag_baseline"):
        validate_sign_cross_lag("nonsense", subject_re_sign_active=True)


def test_vg24_still_validates():
    """The new checks must be inert for a definition that sets no sign lag."""
    validate_model_definition(VG24)
    validate_model_definition(VG25)


# ============================================================
# What the predictor reads
# ============================================================


def _frame(**columns):
    return pd.DataFrame(columns)


def test_the_signed_share_comes_from_the_marginal_and_from_the_cells():
    """Cross-tab rows carry `signed` as NaN and their signed total in two cells.

    Without the cell branch the frame's within-understood cross-tab rows are
    invisible to the lag, which is most of uk_02 and all of uk_07 -- the studies
    that supply the majority of the coefficient's support.
    """
    frame = _frame(
        understood=[100.0, 200.0, 50.0],
        signed=[25.0, np.nan, np.nan],
        signed_only=[np.nan, 30.0, np.nan],
        signed_spoken=[np.nan, 10.0, np.nan],
    )
    signed, understood = sign_share_counts(frame)
    np.testing.assert_array_equal(signed[:2], [25.0, 40.0])
    np.testing.assert_array_equal(understood[:2], [100.0, 200.0])
    # Row 2 has neither source: NaN, not a zero share.
    assert np.isnan(signed[2]) and np.isnan(understood[2])


def test_a_frame_without_cell_columns_still_reads_the_marginal():
    """The typically-developing and bivariate frames carry no cell columns."""
    frame = _frame(understood=[100.0, 80.0], signed=[25.0, np.nan])
    signed, understood = sign_share_counts(frame)
    np.testing.assert_array_equal(signed, [25.0, np.nan])
    np.testing.assert_array_equal(understood, [100.0, np.nan])


def test_a_zero_comprehension_denominator_is_not_a_share():
    """`signed / 0` is undefined, not zero: a wave that understood nothing
    measures no share of it."""
    frame = _frame(understood=[0.0, 10.0], signed=[0.0, 0.0])
    signed, understood = sign_share_counts(frame)
    assert np.isnan(signed[0]) and np.isnan(understood[0])
    # A genuine zero share against a real denominator IS a measurement.
    assert (signed[1], understood[1]) == (0.0, 10.0)


def test_nz01s_produced_cells_are_not_a_source():
    """They partition PRODUCED words, so the only share they support is the
    signed share of production -- a different variable, and pooling the two
    under one coefficient would make it mean two things at once."""
    frame = _frame(
        understood=[np.nan],
        signed=[np.nan],
        signed_only=[np.nan],
        signed_spoken=[np.nan],
        prod_signed_only=[12.0],
        prod_signed_spoken=[8.0],
        prod_total=[40.0],
    )
    signed, understood = sign_share_counts(frame)
    assert np.isnan(signed[0]) and np.isnan(understood[0])


# ============================================================
# The wave walk, shared with VG16's lag
# ============================================================


def test_the_source_is_the_most_recent_strictly_earlier_wave():
    subject = [0, 0, 0]
    age = [12.0, 18.0, 30.0]
    signed = [10.0, 20.0, 30.0]
    understood = [100.0, 100.0, 100.0]
    prev, has_lag, logit = prev_wave_sign_share_lag(subject, age, signed, understood)
    np.testing.assert_array_equal(has_lag, [0.0, 1.0, 1.0])
    np.testing.assert_array_equal(prev, [0, 0, 1])
    assert logit[0] == 0.0
    # The PRIMITIVE's default is the clip, matching VG16's; VG25 registers
    # continuity on the definition, which `prev_wave_sign_share_lag_for_frame`
    # reads. The two are deliberately different and the tests below check both.
    assert logit[1] == pytest.approx(_logit(10 / 100), abs=1e-12)


def test_a_whole_wave_shares_one_source_and_never_its_own():
    """Two forms at one recorded age are one wave, not two (issue #242).

    The row-by-row walk this shares with VG16 advanced state after each row, so
    which of two same-age rows carried the lag depended on arbitrary tie order.
    """
    subject = [0, 0, 0]
    age = [12.0, 24.0, 24.0]
    signed = [10.0, 40.0, 50.0]
    understood = [100.0, 100.0, 200.0]
    prev, has_lag, _ = prev_wave_sign_share_lag(subject, age, signed, understood)
    np.testing.assert_array_equal(has_lag, [0.0, 1.0, 1.0])
    # Both rows of the 24-month wave point at the 12-month wave, neither at the
    # other row of their own.
    np.testing.assert_array_equal(prev, [0, 0, 0])


def test_the_result_does_not_depend_on_the_input_row_order():
    rng = np.random.default_rng(297)
    subject = np.array([0, 0, 0, 1, 1, 2])
    age = np.array([12.0, 24.0, 24.0, 18.0, 30.0, 20.0])
    signed = np.array([5.0, 40.0, 50.0, 3.0, 9.0, 1.0])
    understood = np.array([50.0, 100.0, 200.0, 30.0, 90.0, 10.0])

    reference = prev_wave_sign_share_lag(subject, age, signed, understood)
    for _ in range(8):
        order = rng.permutation(len(subject))
        inverse = np.argsort(order)
        shuffled = prev_wave_sign_share_lag(
            subject[order], age[order], signed[order], understood[order]
        )
        np.testing.assert_array_equal(shuffled[1][inverse], reference[1])
        np.testing.assert_allclose(shuffled[2][inverse], reference[2])


def test_the_largest_denominator_wins_inside_a_source_wave():
    """The least-truncated-measurement rule, said of a ratio.

    A shorter form right-truncates numerator and denominator alike, so the
    wave's largest comprehension total is its least-truncated view of the child
    -- and it is chosen for that reason, not because its share is larger. Here
    the larger denominator carries the SMALLER share, so a rule that maximised
    the share would give a different answer and this would fail.
    """
    subject = [0, 0, 0]
    age = [12.0, 12.0, 24.0]
    signed = [30.0, 40.0, 0.0]
    understood = [60.0, 400.0, 100.0]  # shares 0.50 and 0.10
    prev, has_lag, logit = prev_wave_sign_share_lag(
        subject, age, signed, understood, zero_handling=LAG_ZERO_CLIP
    )
    assert has_lag[2] == 1.0
    assert prev[2] == 1
    assert logit[2] == pytest.approx(_logit(40 / 400), abs=1e-12)


def test_a_wave_with_no_usable_share_is_skipped_rather_than_used():
    subject = [0, 0, 0]
    age = [12.0, 18.0, 24.0]
    signed = [10.0, np.nan, 0.0]
    understood = [100.0, 200.0, 100.0]
    prev, has_lag, _ = prev_wave_sign_share_lag(subject, age, signed, understood)
    # The 18-month wave measures no share, so the 24-month row reaches past it.
    np.testing.assert_array_equal(has_lag, [0.0, 1.0, 1.0])
    np.testing.assert_array_equal(prev, [0, 0, 0])


def test_a_lag_never_crosses_children():
    subject = [0, 1, 1]
    age = [30.0, 12.0, 24.0]
    signed = [50.0, 5.0, 10.0]
    understood = [100.0, 50.0, 80.0]
    _, has_lag, _ = prev_wave_sign_share_lag(subject, age, signed, understood)
    np.testing.assert_array_equal(has_lag, [0.0, 0.0, 1.0])


def test_the_gap_ceiling_drops_the_lag_and_not_the_row():
    subject = [0, 0, 0]
    age = [12.0, 20.0, 45.0]
    signed = [10.0, 20.0, 30.0]
    understood = [100.0, 100.0, 100.0]
    _, unrestricted, _ = prev_wave_sign_share_lag(subject, age, signed, understood)
    prev, restricted, logit = prev_wave_sign_share_lag(
        subject, age, signed, understood, max_gap_months=12.0
    )
    np.testing.assert_array_equal(unrestricted, [0.0, 1.0, 1.0])
    # The 45-month row's source is 25 months back, so it loses its lag; every
    # row is still present, and the source selection is unchanged for the row
    # that keeps one.
    np.testing.assert_array_equal(restricted, [0.0, 1.0, 0.0])
    assert len(prev) == len(logit) == 3
    assert logit[2] == 0.0


# ============================================================
# Boundary shares: the decision measurement forced
# ============================================================


def test_the_clip_gives_every_zero_source_the_same_value():
    """The defect the registered treatment avoids, stated as arithmetic.

    A child who understood 2 words and signed none enters identically to one who
    understood 406 and signed none, at a value fixed by the 1e-4 floor.
    """
    subject = [0, 0, 1, 1]
    age = [12.0, 24.0, 12.0, 24.0]
    signed = [0.0, 5.0, 0.0, 5.0]
    understood = [2.0, 50.0, 406.0, 50.0]
    _, _, logit = prev_wave_sign_share_lag(
        subject, age, signed, understood, zero_handling=LAG_ZERO_CLIP
    )
    assert logit[1] == pytest.approx(-LOGIT_CLIP_HIGH)
    assert logit[3] == pytest.approx(-LOGIT_CLIP_HIGH)
    assert logit[1] == logit[3]


def test_the_continuity_correction_reads_the_waves_own_denominator():
    """The registered treatment. Same two children, two different values."""
    subject = [0, 0, 1, 1]
    age = [12.0, 24.0, 12.0, 24.0]
    signed = [0.0, 5.0, 0.0, 5.0]
    understood = [2.0, 50.0, 406.0, 50.0]
    _, _, logit = prev_wave_sign_share_lag(
        subject, age, signed, understood, zero_handling=LAG_ZERO_CONTINUITY
    )
    assert logit[1] == pytest.approx(_logit(0.5 / 3.0), abs=1e-9)
    assert logit[3] == pytest.approx(_logit(0.5 / 407.0), abs=1e-9)
    # The larger denominator is the stronger evidence of not signing.
    assert logit[3] < logit[1]


def test_a_share_of_exactly_one_is_finite_under_both_treatments():
    """Reachable here and not for VG16's count lag: a child who signs every word
    they understand is a real observation, and the clip bites at both ends."""
    subject = [0, 0]
    age = [12.0, 24.0]
    signed = [40.0, 5.0]
    understood = [40.0, 50.0]
    for treatment, expected in (
        (LAG_ZERO_CLIP, LOGIT_CLIP_HIGH),
        (LAG_ZERO_CONTINUITY, _logit(40.5 / 41.0)),
    ):
        _, _, logit = prev_wave_sign_share_lag(
            subject, age, signed, understood, zero_handling=treatment
        )
        assert np.isfinite(logit[1])
        assert logit[1] == pytest.approx(expected, abs=1e-9)


def test_an_unknown_treatment_raises_from_the_primitive_too():
    with pytest.raises(ValueError, match="sign_lag_zero_handling"):
        prev_wave_sign_share_lag(
            [0, 0], [12.0, 24.0], [1.0, 2.0], [10.0, 20.0], zero_handling="nope"
        )


# ============================================================
# The frame-level entry point reads the definition
# ============================================================


def test_the_frame_entry_point_takes_its_settings_from_the_definition():
    """A caller reaching past it would silently get the registered defaults."""
    frame = pd.DataFrame(
        {
            "subject_code": [0, 0, 0],
            "age": [12.0, 20.0, 45.0],
            "understood": [100.0, 100.0, 100.0],
            "signed": [0.0, 20.0, 30.0],
        }
    )
    _, has_lag, logit = prev_wave_sign_share_lag_for_frame(frame, VG25)
    np.testing.assert_array_equal(has_lag, [0.0, 1.0, 1.0])
    # VG25 registers continuity, so the zero source is not at the clip floor.
    assert logit[1] == pytest.approx(_logit(0.5 / 101.0), abs=1e-9)

    capped = _as_definition_subclass(
        VG25, JointCrossLagModelDefinition, model_id="VG99", sign_lag_max_gap_months=12.0
    )
    _, capped_has_lag, _ = prev_wave_sign_share_lag_for_frame(frame, capped)
    np.testing.assert_array_equal(capped_has_lag, [0.0, 1.0, 0.0])


def test_the_two_lags_select_sources_independently():
    """VG16's reads a count and VG25's a ratio, so a wave usable for one need
    not be usable for the other -- and the shared walk must not conflate them."""
    subject = np.array([0, 0, 0])
    age = np.array([12.0, 18.0, 24.0])
    understood = np.array([100.0, 200.0, 300.0])
    signed = np.array([10.0, np.nan, 30.0])

    _, count_has_lag, _ = prev_wave_lag(subject, age, understood, 810)
    _, share_has_lag, share_prev = prev_wave_sign_share_lag(
        subject, age, signed, np.where(np.isnan(signed), np.nan, understood)
    )
    # The count lag can use the 18-month wave; the share lag cannot.
    np.testing.assert_array_equal(count_has_lag, [0.0, 1.0, 1.0])
    np.testing.assert_array_equal(share_has_lag, [0.0, 1.0, 1.0])
    assert share_prev[2] == pytest.approx(_logit(10 / 100), abs=1e-12)


# ============================================================
# The graph: does the coefficient actually do anything?
# ============================================================
#
# This is the section VG24's LKJ defect argues for. Every check below perturbs
# `beta_sign_lag` and measures what moves, rather than reading the engine and
# concluding that it must.


def _mixed_joint_frame(definition):
    """``synthetic_frame`` with half its children on marginals, not cross-tabs.

    The shared synthetic frame gives **every** row a four-cell partition, so
    ``marginal_outcome_eligible`` is empty and the joint engine builds
    ``y_s_obs`` and ``y_sign_obs`` over zero rows. That is fine for a graph
    fingerprint and useless here: the scope decision this module has to check is
    precisely *which* likelihoods the lag reaches, and a frame with no spoken
    marginal cannot tell the two arms apart.

    Children occupy consecutive row pairs, so clearing the cross-tab columns on
    the second half moves whole children -- both of their waves -- onto the
    marginal path, leaving lags available on each branch.
    """
    frame = synthetic_frame(definition).copy()
    marginal = frame.index >= len(frame) // 2
    for column in (
        "understood_only",
        "signed_only",
        "spoken_only",
        "signed_spoken",
        "cell_total",
        "prod_signed_only",
        "prod_spoken_only",
        "prod_signed_spoken",
        "prod_total",
    ):
        frame.loc[marginal, column] = np.nan
    return frame


def _build(definition, *, output_dir, monkeypatch):
    """Run the joint engine's prior and build stages on :func:`_mixed_joint_frame`.

    A local copy of ``build_synthetic_model`` rather than a call to it, because
    that helper owns the frame and every other caller wants the shared one.
    """
    # The graph render shells out to Graphviz, which is optional and unused
    # here; `.render()` is the only method the engine calls on the result.
    monkeypatch.setattr(
        pymc_utils,
        "model_to_graphviz",
        lambda model: types.SimpleNamespace(render=lambda *a, **k: None),
        raising=False,
    )
    frame = _mixed_joint_frame(definition)
    context = ModelFitContext(
        reporting=reporting.ReportingConfiguration(
            model_name=definition.model_id,
            config_name="synthetic-mixed",
            output_root_dir=output_dir,
            ci_prob=0.89,
            interval_kind="eti",
        ),
        sampling=sampling.SamplingConfiguration(
            draws=4, tune=4, chains=1, cores=1, target_accept=0.8, random_seed=11
        ),
    )
    os.makedirs(context.reporting.output_dir, exist_ok=True)
    context.set_model_data(
        model_data.BinomialModelData(
            X_obs=frame["age"].to_numpy().reshape(-1, 1),
            y_obs=frame["understood"].to_numpy().astype(int),
            n_trials=definition.n_trials,
        ),
        frame,
    )
    _JOINT_ENGINE.resolve("priors")(context, definition)
    _JOINT_ENGINE.resolve("build")(context, definition)
    return context.model


@pytest.fixture(scope="module")
def graphs(tmp_path_factory):
    """VG25, VG24 and three VG25 variants, built once each."""
    from _pytest.monkeypatch import MonkeyPatch

    patcher = MonkeyPatch()
    root = str(tmp_path_factory.mktemp("joint-sign-cross-lag"))
    variants = {
        "vg25": VG25,
        "vg24": VG24,
        "marginal-only": _as_definition_subclass(
            VG25, JointCrossLagModelDefinition, model_id="VG99",
            sign_lag_in_cells=False,
        ),
        "lag-off": _as_definition_subclass(
            VG25, JointCrossLagModelDefinition, model_id="VG99",
            use_sign_cross_lag=False,
        ),
        "population": _as_definition_subclass(
            VG25, JointCrossLagModelDefinition, model_id="VG99",
            sign_lag_baseline="population",
        ),
    }
    try:
        yield {
            name: _build(definition, output_dir=root, monkeypatch=patcher)
            for name, definition in variants.items()
        }
    finally:
        patcher.undo()


def _perturbed(model, *, delta=1.0):
    """``(point, point with beta_sign_lag moved)`` for ``model``."""
    point = model.initial_point()
    key = next(k for k in point if k.startswith("beta_sign_lag"))
    moved = dict(point)
    moved[key] = np.asarray(point[key], dtype=float) + delta
    return point, moved


def _moved_by_beta(model, factor, *, delta=1.0):
    """|change in this factor's log density| when ``beta_sign_lag`` moves.

    An empty likelihood node cannot move, so it would pass every "does not
    depend on beta" assertion for free. :func:`test_the_mixed_frame_exercises_
    every_branch` is what rules that out, and this raises rather than returning
    a comfortable zero if a node ever goes missing.
    """
    rv = model.named_vars[factor]
    point, moved = _perturbed(model, delta=delta)
    logp = model.compile_logp(vars=[rv])
    return abs(float(logp(moved)) - float(logp(point)))


@pytest.mark.slow
@pytest.mark.xdist_group("joint-sign-cross-lag")
def test_the_mixed_frame_exercises_every_branch(graphs):
    """Without this, every "does not move" assertion below could pass vacuously."""
    model = graphs["vg25"]
    for factor in ("y_u_obs", "y_s_obs", "y_sign_obs", "cells_obs"):
        rv = model.named_vars.get(factor)
        assert rv is not None, factor
        assert np.size(rv.tag.observations.data) > 0, f"{factor} has no rows"


@pytest.mark.slow
@pytest.mark.xdist_group("joint-sign-cross-lag")
def test_the_coefficient_is_in_the_graph(graphs):
    """One free RV more than VG24, named, and nothing else added."""
    names25 = [rv.name for rv in graphs["vg25"].free_RVs]
    names24 = [rv.name for rv in graphs["vg24"].free_RVs]
    assert "beta_sign_lag" in names25
    assert set(names25) - set(names24) == {"beta_sign_lag"}
    assert set(names24) - set(names25) == set()


@pytest.mark.slow
@pytest.mark.xdist_group("joint-sign-cross-lag")
def test_the_coefficient_reaches_every_likelihood_it_claims_to(graphs):
    """The check VG24's `+0.000` correlations earned.

    A headline parameter the density does not depend on produces a fit that
    looks exactly like one that found nothing. So this measures the dependence
    rather than asserting it: the spoken marginal and the cross-tab composition
    must move when `beta_sign_lag` moves, and the two likelihoods the lag has no
    business touching must not.
    """
    model = graphs["vg25"]
    for factor in ("y_s_obs", "cells_obs"):
        moved = _moved_by_beta(model, factor)
        assert moved is not None and moved > 1e-8, (
            f"{factor} does not depend on beta_sign_lag"
        )
    # The lag shifts `q`, so comprehension and the signed ratio are untouched.
    for factor in ("y_u_obs", "y_sign_obs"):
        moved = _moved_by_beta(model, factor)
        assert moved == pytest.approx(0.0, abs=1e-10), (
            f"{factor} moved with beta_sign_lag; the lag entered the wrong logit"
        )


@pytest.mark.slow
@pytest.mark.xdist_group("joint-sign-cross-lag")
def test_confining_the_lag_to_the_marginal_takes_it_out_of_the_compositions(graphs):
    """`sign_lag_in_cells=False` must change exactly which factors it reaches.

    This is the registered scope decision expressed as a graph property. If the
    flag did nothing, the `sign-lag-marginal-only` sensitivity would be a second
    fit of the headline under another name -- and it is the arm that says
    whether the headline moved `psi`.
    """
    model = graphs["marginal-only"]
    assert _moved_by_beta(model, "y_s_obs") > 1e-8
    assert _moved_by_beta(model, "cells_obs") == pytest.approx(0.0, abs=1e-10)


@pytest.mark.slow
@pytest.mark.xdist_group("joint-sign-cross-lag")
def test_the_baseline_changes_the_predictor_rather_than_the_factors_it_reaches(graphs):
    """`population` and `within` must reach the same factors and differ inside.

    The population baseline removes the child's signing shift from the
    predictor, which is what makes it the arm in which no estimated per-child
    quantity reaches the cell likelihoods. If the two produced identical
    densities the sensitivity would be measuring nothing.
    """
    within, population = graphs["vg25"], graphs["population"]
    for factor in ("y_s_obs", "cells_obs"):
        assert _moved_by_beta(population, factor) > 1e-8, factor

    _, shifted = _perturbed(within)
    # The two baselines differ by the child's SIGNING shift at the prior wave,
    # which is `tau_subj_sign * z_subj_sign`. At the initial point every `z` is
    # zero, so the shift is zero and the two densities coincide -- correctly,
    # and uselessly. Giving the children distinct signing deviations is what
    # makes the comparison a test of the baseline rather than of nothing.
    z_key = next(k for k in shifted if k.startswith("z_subj_sign"))
    deviations = np.asarray(shifted[z_key], dtype=float)
    shifted[z_key] = np.linspace(-1.5, 1.5, deviations.size).reshape(deviations.shape)

    assert float(within.compile_logp()(shifted)) != pytest.approx(
        float(population.compile_logp()(shifted)), abs=1e-8
    )


@pytest.mark.slow
@pytest.mark.xdist_group("joint-sign-cross-lag")
def test_vg24_is_nested_exactly_at_zero(graphs):
    """At `beta_sign_lag = 0` the observed density must be VG24's, exactly.

    Not approximately: this is what makes "did anything else move?" a red flag
    in the VG25-against-VG24 comparison rather than a vague expectation. Only
    the observed factors are compared, so VG25's extra prior term does not have
    to be accounted for by hand.
    """
    vg25, vg24 = graphs["vg25"], graphs["vg24"]
    point = vg25.initial_point()
    key = next(k for k in point if k.startswith("beta_sign_lag"))
    point[key] = np.zeros_like(np.asarray(point[key], dtype=float))

    shared = {k: v for k, v in point.items() if k != key}
    assert set(shared) == set(vg24.initial_point()), (
        "the two models' free variables differ by more than the coefficient"
    )
    observed_25 = float(vg25.compile_logp(vars=vg25.observed_RVs)(point))
    observed_24 = float(vg24.compile_logp(vars=vg24.observed_RVs)(shared))
    assert observed_25 == pytest.approx(observed_24, rel=1e-12)


@pytest.mark.slow
@pytest.mark.xdist_group("joint-sign-cross-lag")
def test_turning_the_lag_off_reproduces_vg24s_graph(graphs):
    """`use_sign_cross_lag=False` must emit VG24's graph, not VG24's plus `+ 0`.

    The term is added under the flag for this reason: an unconditional
    `+ (term or 0.0)` would put an addition of zero into every joint model's
    `q` logit and move the recorded fingerprint of VG15 and VG24 as well.
    `tests/support/graph_baseline.json` records that it did not; this says why.
    """
    off, vg24 = graphs["lag-off"], graphs["vg24"]
    assert [rv.name for rv in off.free_RVs] == [rv.name for rv in vg24.free_RVs]
    assert [d.name for d in off.deterministics] == [
        d.name for d in vg24.deterministics
    ]
    point = vg24.initial_point()
    assert float(off.compile_logp()(point)) == pytest.approx(
        float(vg24.compile_logp()(point)), rel=1e-12
    )
