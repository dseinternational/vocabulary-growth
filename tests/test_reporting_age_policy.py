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
    max_age_for_sign_ratio,
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

# Sign-bearing ratios of understood (r, p_any, the crossover and their derived
# tables): the tighter of the comprehension and signing caps binds, computed by
# max_age_for_sign_ratio. These stems were mapped to SIGNED until #238, which
# encoded the pre-2026-08-22 assumption that the signing cap was always the
# tighter one -- so this test passed while the artefacts ran to 84 beside a
# policy that says 72.
SIGN_RATIO = "sign_ratio"

# Artefacts whose name carries no outcome suffix, mapped explicitly. Anything
# not listed here and not suffixed is skipped rather than guessed at -- a wrong
# guess would make this test assert the wrong policy and read as a pass.
BY_STEM = {
    "production_rate": ReportedQuantity.RATIO_OF_UNDERSTOOD,
    "production_rate_predictive": ReportedQuantity.RATIO_OF_UNDERSTOOD,
    "production_rate_by_understood": ReportedQuantity.RATIO_OF_UNDERSTOOD,
    "comprehension_production_gap": ReportedQuantity.RATIO_OF_UNDERSTOOD,
    "understood_vs_spoken": ReportedQuantity.RATIO_OF_UNDERSTOOD,
    "understood_vs_spoken_predictive": ReportedQuantity.RATIO_OF_UNDERSTOOD,
    "signed_rate": SIGN_RATIO,
    "sign_speech_crossover": SIGN_RATIO,
    "posterior_summary_r": SIGN_RATIO,
    "posterior_summary_p_any": SIGN_RATIO,
    # VG15's sign-ratio artefacts, previously invisible to this test (no
    # outcome suffix, no stem entry).
    "four_cell_composition": SIGN_RATIO,
    "signing_profile": SIGN_RATIO,
    "p_any_identified_vs_bound": SIGN_RATIO,
    "signed_vs_spoken_rate": SIGN_RATIO,
}


def _cap_for(config, quantity):
    """The reporting cap for a stem's quantity, sign-ratio marker included."""
    if quantity == SIGN_RATIO:
        return max_age_for_sign_ratio(config)
    return max_age_for(config, quantity)

# Artefacts a fitted model writes past its reporting cap, by model id, that the
# policy check is told to excuse. Empty by design: an entry is a stale artefact
# on disk that a fresh fit cannot reproduce, and
# `test_known_stale_entries_are_still_needed` deletes entries the moment a refit
# makes them unnecessary. The VG14 and VG15 84-month sign-ratio and `p_any`
# tables (fitted 2026-08-22 in the gap before the sign-ratio cap followed the
# understood cap) were the last entries, cleared by the 2026-09-01 reporting-
# quality refit (#281).
KNOWN_STALE: dict[str, set[str]] = {}

# Not age-indexed reports: descriptive frames, diagnostics, provenance.
IGNORE_STEMS = {
    "descriptive_statistics",
    "diagnostics",
    "posterior_predictive_calibration",
    "p_any_validation",
    "p_any_validation_gap",
    "joint_trajectory",           # two outcomes; checked separately below
    # Four outcomes with three different caps, so a single-quantity rule cannot
    # describe it. It was mapped to SIGNED until 2026-08-18, which read the file
    # as violating its cap whenever spoken (90) legitimately outran signing (84)
    # -- and, worse, kept VG14's stale-artefact exemption alive after the refit
    # that should have cleared it. Checked by its own test below instead.
    "modality_trajectories",
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
    """Pin the numbers themselves, so a silent edit to the policy is visible.

    These are the caps agreed on 2026-08-22: comprehension (and with it every
    ratio of understood) at 72, signing at 84, spoken at the top of the query
    grid. Until #238 this test pinned the pre-2026-08-22 configuration
    (understood = 84), so it kept passing while several artefacts encoded the
    superseded policy.
    """

    class _DS:
        ages_query = [12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90]
        report_max_age_understood = 72
        report_max_age_signed = 84

    cfg = _DS()
    assert max_age_for(cfg, ReportedQuantity.UNDERSTOOD) == 72
    assert max_age_for(cfg, ReportedQuantity.RATIO_OF_UNDERSTOOD) == 72
    assert max_age_for(cfg, ReportedQuantity.SIGNED) == 84
    assert max_age_for(cfg, ReportedQuantity.SPOKEN) == 90
    # Sign-bearing ratios (r, p_any): whichever of the two caps is tighter binds.
    assert max_age_for_sign_ratio(cfg) == 72


def test_the_registered_ds_signing_models_carry_the_agreed_caps():
    """The registry's own VG14/VG15 definitions, not a synthetic config."""
    for key in ("vg14", "vg15"):
        definition = MODEL_REGISTRY[key]
        assert max_age_for(definition, ReportedQuantity.UNDERSTOOD) == 72
        assert max_age_for(definition, ReportedQuantity.SIGNED) == 84
        assert max_age_for_sign_ratio(definition) == 72


def test_sign_ratio_takes_the_tighter_cap_in_both_directions():
    """The binding rule must survive either cap moving past the other."""

    class _Cfg:
        ages_query = [12, 90]

        def __init__(self, understood, signed):
            self.report_max_age_understood = understood
            self.report_max_age_signed = signed

    assert max_age_for_sign_ratio(_Cfg(72, 84)) == 72
    assert max_age_for_sign_ratio(_Cfg(84, 72)) == 72
    assert max_age_for_sign_ratio(_Cfg(None, 84)) == 84
    assert max_age_for_sign_ratio(_Cfg(None, None)) is None


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
        cap = _cap_for(config, quantity)
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
            label = getattr(quantity, "value", quantity)
            offenders.append(f"{stem}: max age {ages.max():g} > {label} cap {cap:g}")
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
        cap = _cap_for(config, quantity)
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
                label = getattr(quantity, "value", quantity)
                offenders.append(f"{stem}: column {column} > {label} cap {cap:g}")
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
        cap = _cap_for(config, quantity) if quantity else None
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


@needs_fit
@pytest.mark.parametrize(("model_id", "output_dir"), FITTED, ids=[m for m, _ in FITTED])
def test_modality_trajectories_trims_each_series_independently(model_id, output_dir):
    """Four outcomes, three caps — the file a single-quantity rule cannot describe.

    This is the artefact the stem map could not see, and the blind spot cost twice:
    the figure ran to 115 months above a ``p_any`` table trimmed to 84, and then
    the fix for that shipped a CSV whose columns had three different lengths,
    which killed VG14's first refit in the plot stage after 42 minutes of
    sampling.

    Same convention as ``joint_trajectory``: the age column runs to the widest
    cap, and each series is NaN past its own.
    """
    path = os.path.join(output_dir, "modality_trajectories.csv")
    if not os.path.isfile(path):
        pytest.skip("not a trivariate model")
    config = MODEL_REGISTRY[model_id]
    u_cap = max_age_for(config, ReportedQuantity.UNDERSTOOD)
    s_cap = max_age_for(config, ReportedQuantity.SPOKEN)
    sign_cap = max_age_for(config, ReportedQuantity.SIGNED)
    if None in (u_cap, s_cap, sign_cap):
        pytest.skip("model does not cap every modality")

    table = pd.read_csv(path)
    ages = pd.to_numeric(table["age_months"], errors="coerce")
    widest = max(u_cap, s_cap, sign_cap)
    assert ages.max() <= widest + 1e-6, (
        f"modality_trajectories runs to {ages.max():g}, past the widest cap {widest:g}"
    )

    # p_any is a ratio of understood built from the signed ratio: the tighter of
    # the comprehension and signing caps binds (it was min(spoken, signed) until
    # #238, the pre-2026-08-22 components rule).
    any_cap = max_age_for_sign_ratio(config)
    for column, cap in (
        ("understood_median", u_cap),
        ("spoken_median", s_cap),
        ("signed_median", sign_cap),
        ("any_median", any_cap),
    ):
        values = pd.to_numeric(table[column], errors="coerce")
        assert values[ages > cap + 1e-6].isna().all(), (
            f"modality_trajectories reports {column} past {cap:g}"
        )
        assert values[ages <= cap + 1e-6].notna().any(), (
            f"modality_trajectories has no {column} values inside its cap"
        )
