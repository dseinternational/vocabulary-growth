# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Every comprehension- or sign-derived plot must respect its reporting age cap.

This pins a defect class that has now recurred twice. ``report_max_age_understood``
stops a curve where the comprehension evidence stops; ``report_max_age_signed``
does the same for signing. Both were honoured by the *summary tables* and by
``plot_production_rate``, but several plots drawn from the same posterior were
never given the cap, so figures ran to the end of the plot grid while the tables
beside them stopped at the cap.

Two things made that hard to notice by reading:

* The x axis is sometimes a *reparameterisation* of age (expected words
  understood), so the cap is invisible in the plotted coordinates.
* Where the mean is clamped above the upper slope anchor, expected comprehension
  almost stops growing, so the reparameterised axis compresses hard and a gentle
  drift in the extrapolated tail is drawn as a near-vertical step -- which reads
  as a finding about vocabulary rather than an artefact of the transform.

So there are two tests here. The first is behavioural: the cap actually truncates.
The second is structural: every call site in the reporting pipelines passes it,
which is the part a new plot would silently get wrong.
"""

import ast
import inspect
import pathlib

import numpy as np
import pytest

from vocab_growth.models import common_bivariate, common_trivariate

# Call sites that must pass ``max_age_months``. Keyed by the module under test,
# each entry is the plot function's name as called inside the reporting pipeline.
CAPPED_BIVARIATE_CALLS = [
    "plot_production_rate",
    "plot_production_rate_by_understood",
    "plot_production_rate_predictive",
    "plot_comprehension_production_gap",
    "plot_understood_vs_spoken",
    "plot_understood_vs_spoken_predictive",
    "plot_spoken_given_understood",
]

CAPPED_TRIVARIATE_CALLS = [
    "plot_production_rate",
    "plot_signed_rate",
    "plot_sign_speech_crossover",
    "plot_comprehension_production_gap",
]

# Which cap each call site must carry. Passing *a* cap is not enough: until
# 2026-08-13 the trivariate sign-derived figures were passed
# ``report_max_age_understood``, so they satisfied the "is it capped?" test above
# while being trimmed by the wrong outcome's evidence -- and raising the
# comprehension cap from 72 to 84 moved VG14's signed figures as a side effect.
# Keyed by plot function; every other capped call uses the comprehension cap.
SIGN_DERIVED_CALLS = frozenset({"plot_signed_rate", "plot_sign_speech_crossover"})


def _expected_cap_attr(func_name: str) -> str:
    return (
        "report_max_age_signed"
        if func_name in SIGN_DERIVED_CALLS
        else "report_max_age_understood"
    )


def _call_sites(module, func_name):
    """Every ast.Call to ``func_name`` in ``module``'s source."""
    tree = ast.parse(pathlib.Path(inspect.getfile(module)).read_text())
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        name = f.id if isinstance(f, ast.Name) else getattr(f, "attr", None)
        if name == func_name:
            out.append(node)
    return out


@pytest.mark.parametrize(
    ("module", "func_name"),
    [(common_bivariate, n) for n in CAPPED_BIVARIATE_CALLS]
    + [(common_trivariate, n) for n in CAPPED_TRIVARIATE_CALLS],
)
def test_reporting_pipeline_passes_the_age_cap(module, func_name):
    calls = _call_sites(module, func_name)
    assert calls, f"no call to {func_name} found in {module.__name__}"
    for call in calls:
        kwargs = {kw.arg for kw in call.keywords}
        assert "max_age_months" in kwargs, (
            f"{module.__name__} calls {func_name} without max_age_months "
            f"(line {call.lineno}). Comprehension- and sign-derived plots must "
            "stop where their evidence stops; see this module's docstring."
        )


@pytest.mark.parametrize(
    ("module", "func_name"),
    [(common_bivariate, n) for n in CAPPED_BIVARIATE_CALLS]
    + [(common_trivariate, n) for n in CAPPED_TRIVARIATE_CALLS],
)
def test_each_call_site_passes_the_cap_for_its_own_outcome(module, func_name):
    """The cap passed must belong to the outcome plotted.

    A sign-derived figure trimmed by ``report_max_age_understood`` is capped, so
    the test above passes, but it stops where *comprehension* evidence stops and
    moves whenever a comprehension decision is taken. That is the actual defect
    found on VG14 on 2026-08-13.
    """
    expected = _expected_cap_attr(func_name)
    for call in _call_sites(module, func_name):
        kw = next(k for k in call.keywords if k.arg == "max_age_months")
        attr = getattr(kw.value, "attr", None)
        assert attr == expected, (
            f"{module.__name__} calls {func_name} with max_age_months="
            f"...{attr} (line {call.lineno}), expected {expected}. A plot must be "
            "trimmed by its own outcome's reporting cap, not another's."
        )


@pytest.mark.parametrize(
    ("module", "func_name"),
    [(common_bivariate, n) for n in CAPPED_BIVARIATE_CALLS]
    + [(common_trivariate, n) for n in CAPPED_TRIVARIATE_CALLS],
)
def test_capped_plot_functions_accept_the_parameter(module, func_name):
    """The call sites above are only meaningful if the parameter exists."""
    fn = getattr(module, func_name)
    params = inspect.signature(fn).parameters
    assert "max_age_months" in params, f"{func_name} has no max_age_months parameter"
    assert params["max_age_months"].default is None, (
        f"{func_name}'s max_age_months must default to None so an uncapped model "
        "(report_max_age_understood is None) keeps the full grid."
    )


def test_the_cap_actually_truncates_a_reparameterised_axis(tmp_path):
    """``plot_production_rate_by_understood`` is the one that bit us.

    Its x axis is expected comprehension, so a cap expressed in *months* has to be
    applied against ``X_plot`` before the reparameterisation, not after. Build a
    grid where comprehension keeps rising past the cap and check the saved CSV
    stops at the right x value rather than the right row count.
    """
    import pandas as pd

    rng = np.random.default_rng(0)
    n_plot, n_samples, n_trials = 120, 40, 810
    X_plot = np.linspace(8.0, 115.0, n_plot)
    # p_u rises monotonically with age, so words-understood is monotone in age.
    base_u = np.linspace(0.01, 0.85, n_plot)[:, None]
    p_u_plot = np.clip(base_u + rng.normal(0, 0.005, (n_plot, n_samples)), 1e-4, 0.999)
    q_plot = np.clip(
        np.linspace(0.02, 0.9, n_plot)[:, None]
        + rng.normal(0, 0.005, (n_plot, n_samples)),
        1e-4,
        0.999,
    )

    class _Samples:
        pass

    s = _Samples()
    s.X_plot = X_plot
    s.p_u_plot = p_u_plot
    s.q_plot = q_plot

    cap = 72.0
    common_bivariate.plot_production_rate_by_understood(
        s,
        n_trials=n_trials,
        output_dir=str(tmp_path),
        filename="capped",
        max_age_months=cap,
    )
    capped = pd.read_csv(tmp_path / "capped.csv")

    common_bivariate.plot_production_rate_by_understood(
        s, n_trials=n_trials, output_dir=str(tmp_path), filename="uncapped"
    )
    uncapped = pd.read_csv(tmp_path / "uncapped.csv")

    n_expected = int((X_plot <= cap).sum())
    assert len(capped) == n_expected
    assert len(uncapped) == n_plot
    assert len(capped) < len(uncapped)

    # The truncation must happen in age, so the capped curve's last x is the
    # expected comprehension *at the cap* -- not merely a shorter tail.
    expected_last_x = float(np.median(p_u_plot[n_expected - 1, :]) * n_trials)
    assert capped["words_understood"].iloc[-1] == pytest.approx(expected_last_x)
    assert uncapped["words_understood"].max() > capped["words_understood"].max()


def test_the_age_fan_drops_ages_past_the_cap(tmp_path):
    """``plot_spoken_given_understood`` selects representative ages from the query
    grid; the cap has to be applied before that selection, or the fan spends a
    line on an age the model declines to report ``q`` for elsewhere."""
    import pandas as pd

    rng = np.random.default_rng(1)
    ages = np.array([12.0, 30.0, 48.0, 72.0, 90.0])
    q_query = np.clip(
        np.linspace(0.1, 0.8, len(ages))[:, None] + rng.normal(0, 0.01, (len(ages), 50)),
        1e-4,
        0.999,
    )

    class _Samples:
        pass

    s = _Samples()
    s.X_query = ages
    s.q_query = q_query

    common_bivariate.plot_spoken_given_understood(
        s, n_trials=810, output_dir=str(tmp_path), filename="fan", max_age_months=72.0
    )
    fan = pd.read_csv(tmp_path / "fan.csv")
    assert fan["age_months"].max() <= 72
    assert 90 not in set(fan["age_months"].unique())
