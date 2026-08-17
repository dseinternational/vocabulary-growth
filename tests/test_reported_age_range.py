# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Comprehension reporting stops where the comprehension evidence stops.

The query grid is shared by every outcome a model reports, but the outcomes are
not observed over the same ages: in the Down syndrome pool understood has 38
rows at or above 72 months against spoken's 127, and above 84 only 13 against
59. ``report_max_age_understood`` trims understood and ``q`` without touching
spoken.

Two failure modes are specific to this design and neither shows up in a unit
test of the helper alone, so they are tested directly:

* the value reaches the engines through the *configuration* object, not the
  definition, so a missing pass-through raises only when a fit runs;
* trimming must not move a number that is still reported, because it is
  presented as post-processing of an unchanged fit.
"""

import dataclasses

import numpy as np
import pandas as pd
import pytest

import vocab_growth.models.common as common
import vocab_growth.models.common_bivariate as cb
import vocab_growth.models.common_joint_modality as cj
import vocab_growth.models.common_trivariate as ct
from vocab_growth.models import definitions as D
from vocab_growth.posterior_analysis import (
    posterior_summary_table,
    trim_reported_ages,
)

CONFIG_CLASSES = (
    common.ModelConfiguration,
    cb.BivariateModelConfiguration,
    ct.TrivariateModelConfiguration,
    cj.JointModelConfiguration,
)

ENGINE_MODULES = (common, cb, ct, cj)

# The models that report comprehension over a shorter range than production.
TRIMMED_MODELS = ("VG02", "VG05", "VG07", "VG08", "VG09", "VG10", "VG14", "VG15", "VG16")


def _table(ages):
    rng = np.random.default_rng(3)
    draws = rng.beta(2.0, 3.0, size=(len(ages), 64))
    counts = rng.binomial(810, np.clip(draws, 1e-6, 1 - 1e-6))
    return posterior_summary_table(np.asarray(ages, dtype=float), draws, counts, n_trials=810)


# ---------------------------------------------------------------- the helper


def test_trim_drops_only_ages_above_the_cap():
    df = pd.DataFrame({"age_months": [12.0, 48.0, 72.0, 78.0, 90.0], "v": range(5)})
    assert list(trim_reported_ages(df, 72)["age_months"]) == [12.0, 48.0, 72.0]


def test_trim_keeps_the_cap_age_itself():
    """72 is reported, not dropped — it is the last age with evidence, not the
    first without."""
    df = pd.DataFrame({"age_months": [66.0, 72.0, 78.0]})
    assert 72.0 in set(trim_reported_ages(df, 72)["age_months"])


def test_trim_without_a_cap_reports_every_age():
    df = pd.DataFrame({"age_months": [12.0, 90.0]})
    assert list(trim_reported_ages(df, None)["age_months"]) == [12.0, 90.0]


def test_trim_does_not_mutate_its_input():
    df = pd.DataFrame({"age_months": [12.0, 90.0]})
    trim_reported_ages(df, 72)
    assert list(df["age_months"]) == [12.0, 90.0]


def test_trim_reindexes_so_the_frame_reads_as_a_fresh_table():
    df = pd.DataFrame({"age_months": [12.0, 72.0, 90.0]})
    assert list(trim_reported_ages(df, 72).index) == [0, 1]


def test_trim_leaves_every_surviving_value_untouched():
    """The trim is presented as post-processing of an unchanged fit, so a
    reported number must not depend on whether ages above it were dropped."""
    full = _table([12, 24, 48, 72, 78, 90])
    trimmed = trim_reported_ages(full, 72)
    pd.testing.assert_frame_equal(
        trimmed, full[full["age_months"] <= 72].reset_index(drop=True)
    )


# ------------------------------------------------------------- the plumbing


@pytest.mark.parametrize("config_class", CONFIG_CLASSES, ids=lambda c: c.__name__)
def test_every_configuration_class_carries_the_reporting_cap(config_class):
    """``posterior_summary`` reads this off the configuration, not the
    definition. A class missing it raises AttributeError only mid-fit."""
    names = {f.name for f in dataclasses.fields(config_class)}
    assert "report_max_age_understood" in names


@pytest.mark.parametrize("module", ENGINE_MODULES, ids=lambda m: m.__name__.split(".")[-1])
def test_every_engine_passes_the_cap_from_definition_to_configuration(module):
    """Guards the silent half of the failure: a configuration class that has the
    field but an engine that never populates it reports the full grid."""
    source = open(module.__file__).read()
    assert "report_max_age_understood=definition.report_max_age_understood" in source


# ------------------------------------------------------------ the registry


@pytest.mark.parametrize("model_id", TRIMMED_MODELS)
def test_comprehension_reporting_stops_at_84_months(model_id):
    """Raised from 72 to 84 on 2026-08-13. uk_07 and the reinstated uk_06 rows
    rebuilt the older tail, so the 72-84 band carries 25 rows from 20 children
    across five studies rather than the near-nothing the 72 cap was set against.
    84 and no further: above it understood has 13 rows from 11 children, and 84
    is the high trend anchor past which the mean is levelled off, not fitted."""
    definition = getattr(D, model_id)
    assert definition.report_max_age_understood == 84


def test_the_cap_lies_on_the_query_grid():
    """A cap between grid points would silently report one age fewer than it
    names."""
    for model_id in TRIMMED_MODELS:
        definition = getattr(D, model_id)
        assert definition.report_max_age_understood in definition.ages_query


def test_the_cap_actually_removes_query_ages():
    """If the cap were at or above the grid's top it would be inert, and the
    defect it exists to fix would be back without any test failing."""
    for model_id in TRIMMED_MODELS:
        definition = getattr(D, model_id)
        assert definition.report_max_age_understood < max(definition.ages_query)


def test_spoken_only_models_are_left_alone():
    """VG01 is production-only and its data run to 115 months, so trimming it to
    the comprehension range would discard exactly the evidence that argued
    against an 84-month cap."""
    assert D.VG01.report_max_age_understood is None


def test_typically_developing_comprehension_models_carry_a_cap_too():
    """The typically-developing models are NOT exempt, and used not to be tested.

    This test asserted the opposite until 2026-08-17 (#228), on the reasoning
    that "the typically-developing query grids stop at 30 months, well inside
    their data". That is true of *spoken*, which keeps Words & Sentences and does
    reach 30, and false of *understood* by five months: comprehension rides only
    on ``WORDBANK_BIVARIATE_FORMS``, which stop at 25. VG04 and VG12 published a
    27- and a 30-month comprehension median on zero observations for as long as
    the exemption stood.

    Pinning the two current values would only re-pin a premise; the invariant is
    tested against the data in
    ``test_no_model_reports_an_outcome_past_its_last_observation``.
    """
    # Spoken models have no comprehension quantity to trim, and the guard makes
    # setting one an error rather than a silent no-op.
    for model_id in ("VG03", "VG11"):
        assert getattr(D, model_id).report_max_age_understood is None

    # Comprehension models declare one.
    for model_id in ("VG04", "VG12"):
        assert getattr(D, model_id).report_max_age_understood == 25

    # VG13 needs none: its whole observation window stops at 18, so the query
    # grid cannot outrun the evidence in the first place.
    assert D.VG13.report_max_age_understood is None
    assert max(D.VG13.ages_query) == D.VG13.max_age_months == 18


def test_no_model_reports_an_outcome_past_its_last_observation():
    """The policy itself, measured — for every model and every outcome it carries.

    The three defects behind #228 were each a scope rule justified by a property
    of the *pool* that did not hold for one of the outcomes the pool carries.
    Pinned values cannot catch the next one; this sweeps the whole registry and
    compares each outcome's top reported age against the last age at which that
    outcome is actually observed, on the loader's own code path.
    """
    import os

    import vocab_growth.data_utils as vocab_data_utils
    from vocab_growth.reporting_ages import ReportedQuantity, max_age_for

    if not os.path.exists(vocab_data_utils.VOCABULARY_DATA_PATH):
        pytest.skip("prepared vocabulary DuckDB not available")

    ds_pool = vocab_data_utils.load_combined_data()
    quantities = {
        "understood": ReportedQuantity.UNDERSTOOD,
        "spoken": ReportedQuantity.SPOKEN,
        "signed": ReportedQuantity.SIGNED,
    }

    def outcomes_of(definition):
        outcome = getattr(definition, "outcome", None)
        if outcome is not None:
            return [outcome.value]
        if "sign_anchor_ages" in definition.__dataclass_fields__:
            return ["understood", "spoken", "signed"]
        return ["understood", "spoken"]

    offenders = []
    for key, definition in D.MODEL_REGISTRY.items():
        outcomes = outcomes_of(definition)
        if definition.population is D.Population.DOWN_SYNDROME:
            pool = ds_pool
        else:
            pool = vocab_data_utils.load_data(
                D.Population.TYPICALLY_DEVELOPING,
                ["study", "age", *(o for o in outcomes if o != "signed")],
                languages=getattr(definition, "td_languages", D.ENGLISH_LANGUAGES),
                max_age_months=getattr(definition, "max_age_months", None),
            )
            minimum = getattr(definition, "min_study_observations", None)
            if minimum:
                sizes = pool.groupby("study").size()
                pool = pool[pool.study.isin(sizes[sizes >= minimum].index)]
        for outcome in outcomes:
            if outcome not in pool.columns:
                continue
            observed = pool[pool[outcome].notna()]
            if observed.empty:
                continue
            cap = max_age_for(definition, quantities[outcome])
            reported = [
                age for age in definition.ages_query if cap is None or age <= cap
            ]
            beyond = [age for age in reported if age > observed.age.max()]
            if beyond:
                offenders.append(
                    f"{key.upper()} reports {outcome} at {beyond} months, but the "
                    f"last {outcome} observation is at {observed.age.max()}"
                )

    assert not offenders, "\n".join(offenders)


# ------------------------------------------------------------- the guards


def test_a_spoken_model_cannot_carry_a_comprehension_cap():
    """Setting it on VG01 would be a silent no-op, so it is an error instead."""
    with pytest.raises(ValueError, match="comprehension"):
        D.validate_model_definition(
            dataclasses.replace(D.VG01, report_max_age_understood=72)
        )


def test_a_cap_below_the_whole_grid_is_rejected():
    with pytest.raises(ValueError, match="no query age"):
        D.validate_model_definition(
            dataclasses.replace(D.VG10, report_max_age_understood=5)
        )


def test_the_registry_still_validates():
    D.validate_model_registry()
