# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The reporting age policy, checked against a fit's actual output.

``tests/test_reporting_age_caps.py`` checks *call sites* by AST, against a
hand-written list of the plot functions that should be capped. That test cannot
see an artefact nobody thought to cap: on 2026-08-14 sixteen of VG10's tables
ran past their outcome's cap -- fourteen on the full 115-month plot grid and two
more (``posterior_predictive_pmf``/``_cdf``) that a first audit missed entirely,
because those carry age in their *column names* rather than in a column.

So this module checks the other end. Given a fitted model directory it reads
every table, works out which outcome each one reports from its filename, and
asserts the ages respect the policy in :mod:`vocab_growth.reporting_ages`. A new
uncapped artefact fails here the first time a model is fitted, without anyone
having to remember to add it to a list.

The output-directory tests need a fitted model of record and are skipped when
there is none, so the suite still runs on a clean checkout. The policy tests
above them always run.
"""

import glob
import os

import pandas as pd
import pytest

from vocab_growth import environment as env
from vocab_growth.models.definitions import MODEL_REGISTRY
from vocab_growth.reporting_ages import (
    ReportedQuantity,
    max_age_for,
    quantity_for_outcome,
)

# Filename suffix -> the quantity that artefact reports. Order matters: the
# longest suffix must win, or "_sign" is read as "_s".
SUFFIX_QUANTITY = [
    ("_sign", ReportedQuantity.SIGNED),
    ("_u", ReportedQuantity.UNDERSTOOD),
    ("_s", ReportedQuantity.SPOKEN),
    ("_q", ReportedQuantity.RATIO_OF_UNDERSTOOD),
]

# Artefacts whose name carries no outcome suffix, mapped explicitly. Anything
# not listed here and not suffixed is skipped rather than guessed at -- a wrong
# guess would make this test assert the wrong policy and read as a pass.
BY_STEM = {
    "production_rate": ReportedQuantity.RATIO_OF_UNDERSTOOD,
    "production_rate_predictive": ReportedQuantity.RATIO_OF_UNDERSTOOD,
    "production_rate_by_understood": ReportedQuantity.RATIO_OF_UNDERSTOOD,
    "spoken_given_understood": ReportedQuantity.RATIO_OF_UNDERSTOOD,
    "comprehension_production_gap": ReportedQuantity.RATIO_OF_UNDERSTOOD,
    "understood_vs_spoken": ReportedQuantity.RATIO_OF_UNDERSTOOD,
    "understood_vs_spoken_predictive": ReportedQuantity.RATIO_OF_UNDERSTOOD,
    "signed_rate": ReportedQuantity.SIGNED,
    "sign_speech_crossover": ReportedQuantity.SIGNED,
    "posterior_summary_r": ReportedQuantity.SIGNED,
    "posterior_summary_p_any": ReportedQuantity.SIGNED,
    # No outcome suffix, so it matched no rule and was silently skipped -- which
    # is how VG14's modality figure came to run to 115 months directly above a
    # p_any table trimmed to 84. p_any is a union over speaking and signing, so
    # the signing cap is the binding one.
    "modality_trajectories": ReportedQuantity.SIGNED,
}

# Artefacts written by the *summary* stage rather than the plot stage, which
# ``regenerate_plots.py`` cannot refresh because it re-runs the plot stage only.
# When the 84/90 reporting-age policy landed on 2026-08-14 the code that writes
# them was corrected immediately, but the tables already on disk stayed stale
# until their models were refitted; everything else in the family was fixed by
# regeneration alone.
#
# VG15's six entries cleared when its clamp-q-only refit landed at 14:32 on
# 2026-08-14 and VG14's two when its refit landed at 17:11, each caught by
# ``test_known_stale_entries_are_still_needed`` within the hour. Keep the
# mechanism: it is the only thing that distinguishes "this artefact is stale and
# we know it" from "this artefact violates the policy".
#
# The VG14 entry below has a different cause from the summary-stage ones above,
# and it is worth naming. ``modality_trajectories`` *is* a plot-stage artefact,
# so ``regenerate_plots.py`` would normally refresh it -- but VG14's trace was
# written under the ``compact`` persistence tier, and regeneration refuses a
# compacted trace rather than producing silently wrong output. Clearing it
# therefore needs a full refit of VG14, which is deliberately not being done:
# VG14 is superseded by VG15 for every quantity this figure shows, and the
# figure is the ``p_any`` union its own report now warns readers not to use.
# Spending a reporting-quality refit and several gigabytes on a retired model's
# retired figure is the wrong trade; the code is fixed, so the first time VG14
# is refitted for any other reason this entry will fail as unnecessary and
# should be deleted then.
KNOWN_STALE: dict[str, set[str]] = {
    "vg14": {"modality_trajectories"},
}

# Not age-indexed reports: descriptive frames, diagnostics, provenance.
IGNORE_STEMS = {
    "descriptive_statistics",
    "diagnostics",
    "posterior_predictive_calibration",
    "p_any_validation",
    "p_any_validation_gap",
    "joint_trajectory",           # two outcomes; checked separately below
    "joint_trajectory_intervals",
}


def _quantity_for(stem: str) -> ReportedQuantity | None:
    if stem in IGNORE_STEMS:
        return None
    if stem in BY_STEM:
        return BY_STEM[stem]
    for suffix, quantity in SUFFIX_QUANTITY:
        if stem.endswith(suffix) or f"{suffix}_" in stem:
            return quantity
    return None


def _fitted_dirs():
    """(model_id, output_dir) for every model of record present on disk."""
    out = []
    for model_id, definition in MODEL_REGISTRY.items():
        d = os.path.join(
            env.models_output_dir(), f"{definition.model_id}-{definition.config_name}"
        )
        if os.path.isdir(d) and os.path.isfile(os.path.join(d, "fit_manifest.json")):
            out.append((model_id, d))
    return out


FITTED = _fitted_dirs()
needs_fit = pytest.mark.skipif(not FITTED, reason="no fitted model of record on disk")


def test_the_policy_is_the_one_that_was_agreed():
    """Pin the numbers themselves, so a silent edit to the policy is visible."""

    class _DS:
        ages_query = [12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90]
        report_max_age_understood = 84
        report_max_age_signed = 84

    cfg = _DS()
    assert max_age_for(cfg, ReportedQuantity.UNDERSTOOD) == 84
    assert max_age_for(cfg, ReportedQuantity.RATIO_OF_UNDERSTOOD) == 84
    assert max_age_for(cfg, ReportedQuantity.SIGNED) == 84
    assert max_age_for(cfg, ReportedQuantity.SPOKEN) == 90


def test_spoken_tracks_the_query_grid():
    """The spoken cap is derived, so its coupling to ``ages_query`` is the thing
    to pin -- that coupling is what keeps it fingerprint-backed."""

    class _Cfg:
        ages_query = [12, 24, 36]
        report_max_age_understood = 84

    assert max_age_for(_Cfg(), ReportedQuantity.SPOKEN) == 36


def test_a_model_without_caps_reports_the_whole_grid():
    class _TD:
        ages_query = [9, 12, 15, 18, 21, 24, 27, 30]
        report_max_age_understood = None

    td = _TD()
    assert max_age_for(td, ReportedQuantity.UNDERSTOOD) is None
    assert max_age_for(td, ReportedQuantity.RATIO_OF_UNDERSTOOD) is None
    # Spoken still tracks the grid, which for TD is far below any cap anyway.
    assert max_age_for(td, ReportedQuantity.SPOKEN) == 30


def test_outcome_mapping_rejects_the_unexpected():
    with pytest.raises(ValueError):
        quantity_for_outcome("signed")


@needs_fit
@pytest.mark.parametrize(("model_id", "output_dir"), FITTED, ids=[m for m, _ in FITTED])
def test_every_table_respects_its_outcome_cap(model_id, output_dir):
    """No table may carry an age past its own outcome's reporting cap."""
    config = MODEL_REGISTRY[model_id]
    offenders = []
    for path in sorted(glob.glob(os.path.join(output_dir, "*.csv"))):
        stem = os.path.basename(path)[: -len(".csv")]
        quantity = _quantity_for(stem)
        if quantity is None:
            continue
        cap = max_age_for(config, quantity)
        if cap is None:
            continue
        try:
            table = pd.read_csv(path)
        except Exception:  # noqa: BLE001 - unreadable is a different test's problem
            continue
        age_col = next((c for c in ("age_months", "age") if c in table.columns), None)
        if age_col is None:
            continue
        ages = pd.to_numeric(table[age_col], errors="coerce").dropna()
        if len(ages) and ages.max() > cap + 1e-6:
            if stem in KNOWN_STALE.get(model_id, set()):
                continue
            offenders.append(f"{stem}: max age {ages.max():g} > {quantity.value} cap {cap:g}")
    assert not offenders, (
        f"{model_id} reports past its caps:\n  " + "\n  ".join(offenders)
    )


@needs_fit
@pytest.mark.parametrize(("model_id", "output_dir"), FITTED, ids=[m for m, _ in FITTED])
def test_age_keyed_columns_respect_their_cap(model_id, output_dir):
    """``pmf``/``cdf`` carry age in the column names (``pmf_84m``).

    This is the shape a first audit missed, so it gets its own check rather than
    relying on the age-column scan above.
    """
    config = MODEL_REGISTRY[model_id]
    offenders = []
    for path in sorted(glob.glob(os.path.join(output_dir, "*.csv"))):
        stem = os.path.basename(path)[: -len(".csv")]
        if not stem.startswith(("posterior_predictive_pmf", "posterior_predictive_cdf")):
            continue
        quantity = _quantity_for(stem) or quantity_for_outcome(
            MODEL_REGISTRY[model_id].outcome
        )
        cap = max_age_for(config, quantity)
        if cap is None:
            continue
        header = pd.read_csv(path, nrows=0).columns
        for column in header:
            if not column.endswith("m") or "_" not in column:
                continue
            token = column.rsplit("_", 1)[-1][:-1]
            if not token.replace(".", "").isdigit():
                continue
            if float(token) > cap + 1e-6:
                offenders.append(f"{stem}: column {column} > {quantity.value} cap {cap:g}")
    assert not offenders, (
        f"{model_id} has age-keyed columns past its caps:\n  " + "\n  ".join(offenders)
    )


@needs_fit
@pytest.mark.parametrize(("model_id", "output_dir"), FITTED, ids=[m for m, _ in FITTED])
def test_joint_trajectory_trims_each_series_independently(model_id, output_dir):
    """The one figure with two outcomes: understood must stop before spoken."""
    config = MODEL_REGISTRY[model_id]
    u_cap = max_age_for(config, ReportedQuantity.UNDERSTOOD)
    s_cap = max_age_for(config, ReportedQuantity.SPOKEN)
    if u_cap is None or s_cap is None:
        pytest.skip("model does not cap both outcomes")
    for stem in ("joint_trajectory", "joint_trajectory_intervals"):
        path = os.path.join(output_dir, f"{stem}.csv")
        if not os.path.isfile(path):
            continue
        table = pd.read_csv(path)
        ages = pd.to_numeric(table["age_months"], errors="coerce")
        u = pd.to_numeric(table["understood_median"], errors="coerce")
        s = pd.to_numeric(table["spoken_median"], errors="coerce")
        assert ages.max() <= s_cap + 1e-6, f"{stem} runs past the spoken cap"
        assert u[ages > u_cap + 1e-6].isna().all(), (
            f"{stem} reports understood past {u_cap:g}"
        )
        assert s[ages <= s_cap + 1e-6].notna().any(), (
            f"{stem} has no spoken values inside the spoken cap"
        )


@needs_fit
@pytest.mark.parametrize(("model_id", "output_dir"), FITTED, ids=[m for m, _ in FITTED])
def test_known_stale_entries_are_still_needed(model_id, output_dir):
    """A stale-artefact exemption must not outlive the refit that clears it."""
    stale = KNOWN_STALE.get(model_id)
    if not stale:
        pytest.skip("no known-stale artefacts recorded")
    config = MODEL_REGISTRY[model_id]
    unnecessary = []
    for stem in sorted(stale):
        path = os.path.join(output_dir, f"{stem}.csv")
        if not os.path.isfile(path):
            unnecessary.append(f"{stem} (no longer written)")
            continue
        quantity = _quantity_for(stem)
        cap = max_age_for(config, quantity) if quantity else None
        if cap is None:
            continue
        table = pd.read_csv(path)
        age_col = next((c for c in ("age_months", "age") if c in table.columns), None)
        if age_col is None:
            continue
        ages = pd.to_numeric(table[age_col], errors="coerce").dropna()
        if len(ages) and ages.max() <= cap + 1e-6:
            unnecessary.append(f"{stem} (now complies)")
    assert not unnecessary, (
        f"{model_id}: remove these from KNOWN_STALE — they no longer need it:\n  "
        + "\n  ".join(unnecessary)
    )
