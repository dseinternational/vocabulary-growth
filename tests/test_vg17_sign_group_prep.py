# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Regression tests for the VG17/VG18 sign-group modules' data preparation.

VG17 and VG18 are exploratory and unregistered, so nothing in the fit pipeline
validates them. These tests pin the two properties that issue #266 finding 6
found broken: ``_prepare`` used to query ``vocab_combined`` directly and so
applied only two of the DS pool's cleaning rules, and the model carried a study
random effect but no child effect despite most rows coming from children with
repeated visits.
"""

import dataclasses

import numpy as np
import pandas as pd

import vocab_growth.data_utils as data_utils
from vocab_growth.models import model_vg17, model_vg18


def _canonical_keys(outcome):
    """Rows the canonical loader admits for ``outcome`` in VG17's age window."""
    df = data_utils.load_combined_data(include_produced=True)
    df = df[
        df[outcome].notna()
        & df["age"].between(model_vg17.AGE_LO, model_vg17.AGE_HI)
    ]
    return set(
        zip(df["study"], df["subject_id"], df["age"], df[outcome], strict=True)
    )


def test_prepare_rows_come_from_the_canonical_loader():
    """Every prepared row is one the canonical loader admits.

    The old direct-view path admitted 106 extra spoken rows the loader drops
    (100 of them us_01 ceiling-only children), so this is the assertion that
    fails if the module ever goes back to reading the view.
    """
    for outcome in ("spoken", "produced"):
        df, _, _ = model_vg17._prepare(outcome)
        prepared = set(
            zip(df["study"], df["subject_id"], df["age"], df[outcome], strict=True)
        )
        assert prepared <= _canonical_keys(outcome)


def test_prepare_excludes_ceiling_only_children():
    """The rule the direct-view path skipped that moved the most rows."""
    reinstated = data_utils.load_combined_data(
        include_produced=True, include_ceiling_only_children=True
    )
    default = data_utils.load_combined_data(include_produced=True)
    dropped = set(reinstated["study"] + "::" + reinstated["subject_id"].astype(str)) - set(
        default["study"] + "::" + default["subject_id"].astype(str)
    )
    assert dropped, "expected the loader to drop some ceiling-only children"

    df, _, _ = model_vg17._prepare("spoken")
    assert not (set(df["subject_key"]) & dropped)


def test_prepare_keeps_the_model_specific_signing_source_mask():
    """The signing-source rule is VG17's own; the canonical loader does not apply it."""
    df, _, _ = model_vg17._prepare("spoken")
    signed_only = set(data_utils.SIGNED_ONLY_STUDIES) | set(
        data_utils.UNCERTAIN_SIGN_STUDIES
    )
    affected = df[df["study"].isin(signed_only)]
    assert len(affected), "expected the non-comparable signing studies in the pool"
    # Their `signed` is masked, so they can only land in the "unknown" reference.
    assert affected["signed"].isna().all()
    assert (affected["sign_group"] == 0).all()


def test_subject_codes_follow_the_subject_key_convention():
    """`subject_id` is only unique within a study, so codes key on study+subject."""
    df, _, subjects = model_vg17._prepare("spoken")
    expected = df["study"].astype(str) + "::" + df["subject_id"].astype(str)
    pd.testing.assert_series_equal(
        df["subject_key"], expected, check_names=False
    )
    assert subjects == sorted(set(expected))
    assert sorted(df["subject_code"].unique()) == list(range(len(subjects)))
    # The child effect only earns its place if children repeat.
    assert (df.groupby("subject_code").size() > 1).sum() > 0


def test_build_carries_a_child_random_intercept():
    """A study effect alone treats repeated visits as independent observations."""
    pool, _, _ = model_vg17._prepare("spoken")
    # Trim to a couple of studies so the graph stays small; the effect under
    # test is structural, not a function of pool size.
    keep = sorted(pool["study"].unique())[:2]
    df, studies, subjects = model_vg17._prepare("spoken", studies=keep)
    base = model_vg17._config()
    config = dataclasses.replace(
        base,
        # VG01's query ages run past VG17's 12-66 month window; clip them so this
        # structural test does not depend on the grid's domain check.
        ages_query=[
            a for a in base.ages_query if model_vg17.AGE_LO <= a <= model_vg17.AGE_HI
        ],
    )
    model, _ = model_vg17._build(
        df, studies, config, y_col="spoken", subjects=subjects
    )
    names = {v.name for v in model.free_RVs}
    assert "tau_subj" in names
    assert "delta_subj_raw" in names
    assert len(model.coords["subject_id"]) == len(subjects)
    # The project's convention for a child scale.
    assert model_vg17.TAU_SUBJECT_SIGMA == 1.5


def test_vg18_caution_is_prominent_and_names_the_mechanism():
    """The signer contrast is partly an identity; that must be impossible to miss."""
    assert "CAUTION" in model_vg18.__doc__
    assert "CAUTION" in model_vg18.CAUTION
    lowered = model_vg18.CAUTION.lower()
    assert "mechanical" in lowered
    assert "signed" in lowered and "produced" in lowered
    assert "descriptive" in lowered


def test_vg18_sign_group_is_derived_from_a_component_of_its_outcome():
    """Pins the fact the caution describes, so it cannot become stale silently."""
    df, _, _ = model_vg17._prepare("produced")
    union = df[df["signed"].notna() & (df["signed"] > 0)]
    assert len(union), "expected signers in the produced pool"
    # For a signer, `produced` is at least `spoken` and generally exceeds it:
    # the sign group is defined by words that also enter the outcome.
    assert (union["produced"] >= union["spoken"]).all()
    assert np.any(union["produced"] > union["spoken"])
