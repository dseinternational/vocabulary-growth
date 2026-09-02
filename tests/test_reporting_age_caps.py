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
# The 2026-08-13 fix then over-corrected: it mapped those figures to
# ``report_max_age_signed`` alone, and when the comprehension cap moved DOWN to
# 72 on 2026-08-22 the signing cap stopped being the tighter one, so the
# figures ran to 84 against a policy that says 72 (#238). The trivariate
# pipeline now computes named cap locals from vocab_growth.reporting_ages --
# ``ratio_cap`` for ratios of understood, ``sign_ratio_cap`` (the tighter of
# the comprehension and signing caps) for the sign-bearing ratios -- and each
# call site must pass the local named for its quantity. The bivariate pipeline
# still passes the definition attribute directly; its one cap is the
# comprehension cap, so there is no wrong attribute to pick.
#
# Values: ("attr", name) for an attribute access, ("name", name) for a local.
EXPECTED_CAP = {
    (common_bivariate, "plot_production_rate"): ("attr", "report_max_age_understood"),
    (common_bivariate, "plot_production_rate_by_understood"): ("attr", "report_max_age_understood"),
    (common_bivariate, "plot_production_rate_predictive"): ("attr", "report_max_age_understood"),
    (common_bivariate, "plot_comprehension_production_gap"): ("attr", "report_max_age_understood"),
    (common_bivariate, "plot_understood_vs_spoken"): ("attr", "report_max_age_understood"),
    (common_bivariate, "plot_understood_vs_spoken_predictive"): ("attr", "report_max_age_understood"),
    (common_trivariate, "plot_production_rate"): ("name", "ratio_cap"),
    (common_trivariate, "plot_comprehension_production_gap"): ("name", "ratio_cap"),
    (common_trivariate, "plot_signed_rate"): ("name", "sign_ratio_cap"),
    (common_trivariate, "plot_sign_speech_crossover"): ("name", "sign_ratio_cap"),
}


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
    kind, expected = EXPECTED_CAP[(module, func_name)]
    for call in _call_sites(module, func_name):
        kw = next(k for k in call.keywords if k.arg == "max_age_months")
        if kind == "attr":
            actual = getattr(kw.value, "attr", None)
        else:
            actual = kw.value.id if isinstance(kw.value, ast.Name) else None
        assert actual == expected, (
            f"{module.__name__} calls {func_name} with max_age_months="
            f"...{actual} (line {call.lineno}), expected {expected}. A plot must be "
            "trimmed by its own quantity's reporting cap, not another's."
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


def test_modality_trajectory_csv_shares_one_age_grid(tmp_path):
    """The per-outcome caps must not give the CSV columns of different lengths.

    A third instance of the same defect class, found the hard way: VG14's first
    refit after per-outcome caps arrived sampled for 40 minutes and then died in
    the plot stage with ``ValueError: All arrays must be of the same length``.
    The CSV paired the full ``X_plot`` age column with median arrays trimmed at
    three different caps (understood 84, signed 84, spoken 90).

    Two things hid it. The figure is written *before* the CSV and draws each
    curve against its own trimmed x, so the plot was always correct. And
    ``modality_trajectories`` carries no outcome suffix, so it matches no stem in
    the reporting-age policy test's map -- the same blind spot that let the
    figure run to 115 months in the first place.

    The fix keeps one shared age column and masks past each cap with NaN, so the
    CSV says "not reported here" rather than silently realigning rows.
    """
    import types

    import numpy as np
    import pandas as pd

    from vocab_growth.models import common_trivariate as ct

    ages = np.arange(8.0, 116.0, 1.0)
    n_draws = 40
    rng = np.random.default_rng(0)

    def grid():
        return rng.uniform(0.05, 0.5, size=(ages.size, n_draws))

    samples = types.SimpleNamespace(
        X_plot=ages,
        p_u_plot=grid(),
        p_s_plot=grid(),
        p_sign_plot=grid(),
        p_any_plot=grid(),
    )

    ct.plot_modality_trajectories(
        samples,
        n_trials=810,
        output_dir=str(tmp_path),
        filename="modality_trajectories",
        max_age_months_understood=72,
        max_age_months_spoken=90,
        max_age_months_signed=84,
        # p_any's cap is explicit since #238: the call site passes
        # reporting_ages.max_age_for_sign_ratio (tighter of comprehension and
        # signing), rather than the function deriving min(spoken, signed).
        max_age_months_any=72,
    )

    frame = pd.read_csv(tmp_path / "modality_trajectories.csv")
    # One shared age column, trimmed to the WIDEST cap -- ``joint_trajectory``'s
    # convention. The file then carries no row that no series reports on, and no
    # column implying a series was reported where it was not.
    widest = 90
    assert frame.age_months.max() == widest
    assert len(frame) == int((ages <= widest).sum()), "one row per in-range grid age"

    caps = {
        "understood_median": 72,
        "spoken_median": 90,
        "signed_median": 84,
        "any_median": 72,
        "any_ci_lo": 72,
        "any_ci_hi": 72,
    }
    for column, cap in caps.items():
        inside = frame.loc[frame.age_months <= cap, column]
        outside = frame.loc[frame.age_months > cap, column]
        assert inside.notna().all(), f"{column} is missing values inside its cap"
        assert outside.isna().all(), f"{column} reports past its {cap}-month cap"


def test_the_age_fan_is_retired():
    """``plot_spoken_given_understood`` drew a population rate as a straight line to
    810 words -- E[q | U] by construction, the reading issue #233 rules out --
    and was retired on 2026-09-02 in favour of ``plot_understood_vs_spoken``
    carrying the observed children."""
    assert not hasattr(common_bivariate, "plot_spoken_given_understood")


def _stub_samples(*, with_observed: bool):
    rng = np.random.default_rng(3)
    n_plot, n_samples = 60, 30
    X_plot = np.linspace(8.0, 72.0, n_plot)
    p_u = np.clip(np.linspace(0.02, 0.6, n_plot)[:, None] + rng.normal(0, 0.003, (n_plot, n_samples)), 1e-4, 0.999)
    q = np.clip(np.linspace(0.02, 0.8, n_plot)[:, None] + rng.normal(0, 0.003, (n_plot, n_samples)), 1e-4, 0.999)

    class _S:
        pass

    s = _S()
    s.X_plot = X_plot
    s.p_u_plot = p_u
    s.q_plot = q
    s.p_s_plot = p_u * q
    s.y_u_plot = np.round(p_u * 810)
    s.y_s_plot = np.round(p_u * q * 810)
    if with_observed:
        n_obs = 400
        s.X_obs = rng.uniform(8, 72, n_obs)
        u = rng.integers(0, 700, n_obs).astype(float)
        u[:20] = np.nan
        s.y_u_obs = u
        s.y_s_obs = np.where(np.isnan(u), np.nan, np.round(u * rng.uniform(0.05, 0.6, n_obs)))
    return s


@pytest.mark.parametrize("func", ["plot_understood_vs_spoken", "plot_production_rate_by_understood",
                                  "plot_understood_vs_spoken_predictive"])
def test_the_observed_overlay_is_optional(tmp_path, func):
    """The age-cap tests build stubs without observed counts; the overlay must not
    demand them, and must write its own CSV only when it drew something."""
    import pandas as pd

    plot = getattr(common_bivariate, func)
    plot(_stub_samples(with_observed=False), n_trials=810, output_dir=str(tmp_path), filename="bare")
    assert (tmp_path / "bare.csv").exists()
    assert not (tmp_path / "bare_observed.csv").exists()

    plot(_stub_samples(with_observed=True), n_trials=810, output_dir=str(tmp_path), filename="obs")
    if func == "plot_understood_vs_spoken_predictive":
        return  # the predictive figure draws the observed points but summarises nothing
    table = pd.read_csv(tmp_path / "obs_observed.csv")
    assert list(table.columns) == ["level", "n", "median", "q25", "q75"]
    assert (table["n"] >= 10).all() and (table["q25"] <= table["median"]).all()
    if func == "plot_production_rate_by_understood":
        assert table["median"].between(0, 1).all()
