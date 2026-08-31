# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""VG14 and VG15 can now run the marginal-fallback sensitivity (issue #266 finding 8).

Roughly 455 of 1,428 Down syndrome spoken rows cannot condition on an observed
comprehension count, and take ``BB(810, p_U*q, kappa)`` instead. That is
mean-correct but is **not** the marginal implied by the paired model — it misses
the variance — and the affected rows are older and clustered by study, so the
approximation is not ignorable. The bivariate engines have carried a choice of
treatment since #240; the trivariate and joint engines hard-coded the default, so
no sensitivity could be run at all on the two signing models.

Both now route their child-outcome likelihoods through the same shared builder,
for the **signed** rows as well as the spoken ones: signing is nested inside
comprehension exactly as speech is, so exposing the choice for one outcome and
not the other would leave half the exposure unmeasurable.

Two properties matter and are checked separately, because a routing change that
satisfied only the first would be decorative:

* under the default, the graph is **unchanged** —
  ``tests/test_graph_equivalence.py`` pins that for all twenty models;
* under each alternative, the graph **actually differs**, on a frame that has
  marginal rows to differ on. The synthetic frame the equivalence harness uses
  is fully paired, so ``paired_only`` and ``moment_matched`` are no-ops there;
  a test that used it would pass while proving nothing.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from support.synthetic_graphs import (
    build_synthetic_model,
    fixed_point_logp,
    synthetic_frame,
)

from vocab_growth.models.catalogue import get
from vocab_growth.models.definitions import VG14, VG15
from vocab_growth.models.fit_identity import BACKFILL_DEFAULTS, definition_differences
from vocab_growth.models.likelihood_utils import (
    SPOKEN_FALLBACK_MOMENT_MATCHED,
    SPOKEN_FALLBACK_PAIRED_ONLY,
    SPOKEN_FALLBACK_PRODUCT,
    SPOKEN_FALLBACK_SEPARATE_DISPERSION,
    SPOKEN_FALLBACK_TREATMENTS,
)

pytestmark = pytest.mark.slow

_MODELS = (VG14, VG15)


def _frame_with_marginal_rows(definition, *, n_marginal=8):
    """A frame where some rows record speech and signing but no comprehension.

    Those are the rows every treatment differs on. The harness's own frame is
    fully paired, so it cannot distinguish them.
    """
    frame = synthetic_frame(definition).copy()
    rows = frame.index[:n_marginal]
    frame.loc[rows, "understood"] = np.nan
    # A row with no comprehension count cannot carry a within-understood
    # composition either, so it leaves the cross-tabulation entirely. Clearing
    # only some cells would leave the joint engine reading NaN counts.
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
        if column in frame:
            frame.loc[rows, column] = np.nan
    return frame


def _logp(definition, frame, tmp_path, monkeypatch):
    engine = get(definition.model_id.lower()).engine
    context = build_synthetic_model(
        definition, engine, output_dir=str(tmp_path), monkeypatch=monkeypatch
    )
    # The frame is injected by rebuilding with it in place of the default.
    return context, fixed_point_logp(context.model)


@pytest.mark.parametrize("base", _MODELS, ids=lambda d: d.model_id)
def test_the_definition_carries_the_choice(base):
    assert base.spoken_fallback == SPOKEN_FALLBACK_PRODUCT
    assert base.spoken_fallback_kappa_sigma == 0.5
    for treatment in SPOKEN_FALLBACK_TREATMENTS:
        variant = dataclasses.replace(base, spoken_fallback=treatment)
        assert variant.spoken_fallback == treatment


@pytest.mark.parametrize("base", _MODELS, ids=lambda d: d.model_id)
def test_an_unknown_treatment_is_refused(base, tmp_path, monkeypatch):
    engine = get(base.model_id.lower()).engine
    variant = dataclasses.replace(base, spoken_fallback="not_a_treatment")
    with pytest.raises(ValueError, match="Unknown spoken_fallback"):
        build_synthetic_model(
            variant, engine, output_dir=str(tmp_path), monkeypatch=monkeypatch
        )


@pytest.mark.parametrize("base", _MODELS, ids=lambda d: d.model_id)
def test_separate_dispersion_adds_one_offset_per_nested_outcome(
    base, tmp_path, monkeypatch
):
    """Spoken and signed each get their own, because each has its own rows."""
    engine = get(base.model_id.lower()).engine
    variant = dataclasses.replace(
        base, spoken_fallback=SPOKEN_FALLBACK_SEPARATE_DISPERSION
    )
    context = build_synthetic_model(
        variant, engine, output_dir=str(tmp_path), monkeypatch=monkeypatch
    )
    names = {rv.name for rv in context.model.free_RVs}
    assert "log_kappa_s_fallback" in names
    assert "log_kappa_sign_fallback" in names

    default = build_synthetic_model(
        base, engine, output_dir=str(tmp_path / "default"), monkeypatch=monkeypatch
    )
    baseline = {rv.name for rv in default.model.free_RVs}
    assert names - baseline == {"log_kappa_s_fallback", "log_kappa_sign_fallback"}


@pytest.mark.parametrize("base", _MODELS, ids=lambda d: d.model_id)
def test_every_treatment_changes_the_likelihood_on_rows_that_have_one(
    base, tmp_path, monkeypatch
):
    """The routing must be substantive, not decorative.

    Checked on a frame that HAS marginal rows. On the fully paired frame the
    equivalence harness uses, `paired_only` drops nothing and `moment_matched`
    never selects its branch, so all four treatments agree there — a test that
    used it would pass while proving nothing.
    """
    import dse_research_utils.statistics.models.data as model_data

    engine = get(base.model_id.lower()).engine
    frame = _frame_with_marginal_rows(base)

    scores = {}
    for treatment in SPOKEN_FALLBACK_TREATMENTS:
        variant = dataclasses.replace(base, spoken_fallback=treatment)
        context = build_synthetic_model(
            variant,
            engine,
            output_dir=str(tmp_path / treatment),
            monkeypatch=monkeypatch,
        )
        # Rebuild against the marginal-row frame rather than the default one.
        context.set_model_data(
            model_data.BinomialModelData(
                X_obs=frame["age"].to_numpy().reshape(-1, 1),
                y_obs=frame["understood"].fillna(0).to_numpy().astype(int),
                n_trials=variant.n_trials,
            ),
            frame,
        )
        engine.resolve("priors")(context, variant)
        engine.resolve("build")(context, variant)
        scores[treatment] = fixed_point_logp(context.model)

    # Every treatment is a distinct likelihood on these rows.
    assert len(set(scores.values())) == len(SPOKEN_FALLBACK_TREATMENTS), (
        f"{base.model_id}: treatments do not all differ — {scores}"
    )
    # And paired-only drops the marginal rows, so it scores strictly fewer of them.
    assert scores[SPOKEN_FALLBACK_PAIRED_ONLY] != scores[SPOKEN_FALLBACK_PRODUCT]
    assert scores[SPOKEN_FALLBACK_MOMENT_MATCHED] != scores[SPOKEN_FALLBACK_PRODUCT]


# --- and adding the field invalidated nothing -----------------------------------


@pytest.mark.parametrize("base", _MODELS, ids=lambda d: d.model_id)
def test_a_fit_made_before_the_field_existed_still_validates(base):
    """The first real use of the backfill mechanism (#273 step 5).

    Under raw dictionary equality, adding this field would have invalidated
    every VG14 and VG15 fit ever made, for a field whose default is what those
    fits already did.
    """
    from vocab_growth.fit_artifacts import normalise_for_json

    recorded = normalise_for_json(base)
    for field in ("spoken_fallback", "spoken_fallback_kappa_sigma"):
        assert field in BACKFILL_DEFAULTS
        recorded.pop(field)
    assert definition_differences(recorded, base) == []


@pytest.mark.parametrize("base", _MODELS, ids=lambda d: d.model_id)
def test_the_backfill_does_not_excuse_a_real_change(base):
    """It says what the absence meant, not that the field may be anything."""
    from vocab_growth.fit_artifacts import normalise_for_json

    recorded = normalise_for_json(base)
    recorded.pop("spoken_fallback")
    altered = dataclasses.replace(base, spoken_fallback=SPOKEN_FALLBACK_PAIRED_ONLY)
    differences = definition_differences(recorded, altered)
    assert any(d.field == "spoken_fallback" for d in differences)


def test_the_backfill_claim_matches_what_the_resolver_actually_did():
    """The entry is a claim about history; this is the code that made it true."""
    from vocab_growth.models.likelihood_utils import resolve_fallback_treatment

    class _WithoutTheField:
        pass

    assert (
        resolve_fallback_treatment(_WithoutTheField())
        == BACKFILL_DEFAULTS["spoken_fallback"]
        == SPOKEN_FALLBACK_PRODUCT
    )
