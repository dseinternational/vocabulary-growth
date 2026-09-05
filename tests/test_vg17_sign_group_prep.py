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


import numpy as np
import pandas as pd

import vocab_growth.data_utils as data_utils
from vocab_growth.models.exploratory import vg17 as model_vg17
from vocab_growth.models.exploratory import vg18 as model_vg18


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
    # The public configuration, unmodified. It used to be rewritten here --
    # VG01's query ages run past this model's 12-66 month window, so the grid's
    # domain check refused the default build -- and that workaround was the only
    # reason the test passed while `fit()` could not run at all. `_config()`
    # now clips the grid at source (issue #273 finding 4), so a test that has to
    # repair the configuration would mean the repair had come undone.
    model, _ = model_vg17._build(
        df, studies, model_vg17._config(), y_col="spoken", subjects=subjects
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


# --- a produced union with no separable sign component ---------------------------


def test_vg18_excludes_a_produced_union_with_no_separable_sign_component():
    """us_03 cannot be placed in a sign group, so it must not enter VG18.

    Its expressive cell is "understands and says **or signs**" with no separable
    sign count, so every row would be grouped `unknown` while its outcome
    contains the exposure the model contrasts on. That is not the same fault
    VG18's docstring already warns about: the union studies record `signed`, so
    they are grouped correctly and their confound is visible in the contrast.
    """
    df, _, _ = model_vg17._prepare("produced")
    assert "us_03" not in set(df["study"])


def test_the_spoken_outcome_is_not_touched_by_that_exclusion():
    """The rule is about the produced union, not about the study.

    VG17's `spoken` outcome does not contain `signed` -- which is why it is the
    interpretable contrast -- so a source is only excluded where its union hides
    the sign component. us_03 reaches neither model, but for different reasons:
    here it simply has no spoken count to pass the outcome filter.
    """
    spoken, _, _ = model_vg17._prepare("spoken")
    assert "us_03" not in set(spoken["study"])

    pool = data_utils.load_combined_data(include_produced=True)
    us03 = pool[pool["study"] == "us_03"]
    assert len(us03) > 0, "us_03 is in the pool; this test would be vacuous otherwise"
    assert us03["spoken"].isna().all(), "excluded by the outcome filter, not the rule"


def test_the_ungroupable_union_rule_names_its_sources_and_is_reversible():
    frame = pd.DataFrame(
        {
            "study": ["us_03", "us_03", "uk_02"],
            "subject_id": ["a", "b", "c"],
            "age": [20.0, 30.0, 30.0],
            "produced": [5.0, 40.0, 60.0],
        }
    )
    kept, dropped = data_utils.drop_ungroupable_produced_unions(frame)
    assert set(kept["study"]) == {"uk_02"}
    assert dropped == {"us_03": 2}

    reinstated, none_dropped = data_utils.drop_ungroupable_produced_unions(
        frame, include_ungroupable_produced_unions=True
    )
    assert len(reinstated) == 3
    assert none_dropped == {}


def test_the_ungroupable_union_registry_is_not_empty_and_holds_registry_studies():
    """An empty registry would make the rule silently inert."""
    registry = data_utils.PRODUCED_UNION_WITHOUT_SIGN_DETAIL
    assert registry
    pool = data_utils.load_combined_data(include_produced=True)
    assert registry <= set(pool["study"]), "names a study the pool does not carry"
