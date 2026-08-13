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


def test_typically_developing_models_are_left_alone():
    """The typically-developing query grids stop at 30 months, well inside their
    data; the asymmetry being corrected here is specific to the Down syndrome
    pool."""
    for model_id in ("VG03", "VG04", "VG11", "VG12", "VG13"):
        assert getattr(D, model_id).report_max_age_understood is None


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
