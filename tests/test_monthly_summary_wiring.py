# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Every engine's monthly-summary call site must survive a real invocation.

The wiring passes engine-specific sample arrays into a shared helper, and a
wrong attribute name there raises only when the fit actually runs — invisible to
a unit test that exercises the helper alone. That is the failure mode that got
through review once already in this codebase, so these tests reach into each
engine's own sample class and call the emitter for real.
"""

import inspect

import numpy as np
import pandas as pd
import pytest

import vocab_growth.models.common as common
import vocab_growth.models.common_bivariate as cb
import vocab_growth.models.common_joint_modality as cj
import vocab_growth.models.common_trivariate as ct

N_SAMPLES = 64
N_PLOT = 500


def _grid():
    rng = np.random.default_rng(11)
    X_plot = np.linspace(11.0, 86.0, N_PLOT)
    p = 1.0 / (1.0 + np.exp(-(X_plot - 45.0) / 12.0))
    p_plot = np.repeat(p[:, None], N_SAMPLES, axis=1)
    y_plot = rng.binomial(810, np.clip(p_plot, 1e-6, 1 - 1e-6))
    return X_plot, p_plot, y_plot


def test_emit_monthly_summary_writes_table_and_figure(tmp_path):
    X_plot, p_plot, y_plot = _grid()
    monthly = common.emit_monthly_summary(
        output_dir=str(tmp_path),
        X_plot=X_plot,
        p_plot=p_plot,
        y_plot=y_plot,
        X_obs=pd.Series([12.0, 12.0, 24.0]),
        n_trials=810,
        ci_prob=0.89,
        suffix="u",
        outcome_label="words understood",
        y_label="Predicted words understood",
    )
    written = {path.name for path in tmp_path.iterdir()}
    assert "posterior_summary_monthly_u.csv" in written
    assert "expected_counts_by_month_u.png" in written
    assert "expected_counts_by_month_u.svg" in written
    # The figure gets no sidecar CSV: it is drawn from the frame already written
    # under the canonical name, so a sidecar would be a byte-identical second
    # copy of it under a second name.
    assert "expected_counts_by_month_u.csv" not in written
    reloaded = pd.read_csv(tmp_path / "posterior_summary_monthly_u.csv")
    assert len(reloaded) == len(monthly)
    assert reloaded["age_months"].tolist() == monthly["age_months"].tolist()


def test_emit_monthly_summary_handles_absent_predictive_draws(tmp_path):
    X_plot, p_plot, _ = _grid()
    common.emit_monthly_summary(
        output_dir=str(tmp_path),
        X_plot=X_plot,
        p_plot=p_plot,
        y_plot=None,
        X_obs=pd.Series([12.0]),
        n_trials=810,
        ci_prob=0.89,
        suffix="sign",
        outcome_label="words signed",
        y_label="Expected words signed",
    )
    assert (tmp_path / "posterior_summary_monthly_sign.csv").exists()
    assert (tmp_path / "expected_counts_by_month_sign.png").exists()


@pytest.mark.parametrize(
    "helper, samples_class, outcomes",
    [
        (cb._run_bivariate_outcome_plots, cb.BivariateModelSamples, ("u", "s")),
        (ct._run_trivariate_outcome_plots, ct.TrivariateModelSamples, ("u", "s", "sign")),
    ],
)
def test_outcome_plot_helpers_accept_the_p_plot_argument(helper, samples_class, outcomes):
    """The shared helpers must take p_plot, and the samples must supply it.

    Guards the two halves of the wiring separately: the helper's signature, and
    the existence of the ``p_<outcome>_plot`` field each call site reads.
    """
    assert "p_plot" in inspect.signature(helper).parameters
    fields = set(samples_class.__dataclass_fields__)
    for outcome in outcomes:
        assert f"p_{outcome}_plot" in fields
        assert f"y_{outcome}_plot" in fields


def test_joint_engine_reads_only_fields_its_samples_have():
    """The joint engine's monthly block must not name a sample field it lacks.

    It has no ``y_*_plot``, which is why it passes ``y_plot=None``; it does need
    ``p_u_plot``, ``q_plot`` and ``r_plot``.
    """
    fields = set(cj.JointModelSamples.__dataclass_fields__)
    assert {"X_plot", "p_u_plot", "q_plot", "r_plot"} <= fields
    assert not [name for name in fields if name.startswith("y_") and name.endswith("_plot")]


def test_n_obs_counts_only_the_administrations_observing_that_outcome(tmp_path):
    """n_obs is per-outcome, not per-administration.

    A first cut of the joint-engine wiring passed the whole analysis frame's
    ages for all three outcomes, so every outcome reported the same total and
    overstated the coverage of the sparser ones. The other engines pass their
    per-outcome ``x_obs``; this pins the contract the shared emitter relies on.
    """
    X_plot, p_plot, y_plot = _grid()
    frame = pd.DataFrame(
        {
            "age": [12.0, 12.0, 12.0, 24.0, 24.0],
            "understood": [10.0, 20.0, 30.0, 40.0, 50.0],
            "signed": [1.0, np.nan, np.nan, 2.0, np.nan],
        }
    )

    understood = common.emit_monthly_summary(
        output_dir=str(tmp_path),
        X_plot=X_plot,
        p_plot=p_plot,
        y_plot=y_plot,
        X_obs=frame.loc[frame["understood"].notna(), "age"],
        n_trials=810,
        ci_prob=0.89,
        suffix="u",
        outcome_label="words understood",
    )
    signed = common.emit_monthly_summary(
        output_dir=str(tmp_path),
        X_plot=X_plot,
        p_plot=p_plot,
        y_plot=None,
        X_obs=frame.loc[frame["signed"].notna(), "age"],
        n_trials=810,
        ci_prob=0.89,
        suffix="sign",
        outcome_label="words signed",
    )

    assert understood.set_index("age_months")["n_obs"].loc[12] == 3
    assert signed.set_index("age_months")["n_obs"].loc[12] == 1
    # The sparser outcome must not inherit the denser one's coverage.
    assert int(signed["n_obs"].sum()) == 2
    assert int(understood["n_obs"].sum()) == 5


def test_joint_engine_scopes_n_obs_per_outcome():
    """The joint engine's monthly block must filter its X_obs by outcome column."""
    source = inspect.getsource(cj.posterior_summary)
    assert "emit_monthly_summary" in source
    # It must select on the outcome column rather than handing over the frame.
    assert 'analysis_df[column].notna()' in source
    assert 'X_obs=context.analysis_df["age"]' not in source

# --------------------------------------------------------------------------
# No engine may rely on the emitter's interval-kind default
# --------------------------------------------------------------------------

_ENGINE_MODULES_WITH_MONTHLY = (common, cb, ct, cj)


@pytest.mark.parametrize(
    "module", _ENGINE_MODULES_WITH_MONTHLY, ids=lambda m: m.__name__.rpartition(".")[2]
)
def test_every_emit_monthly_summary_call_passes_interval_kind(module):
    """`emit_monthly_summary` defaults `interval_kind` to "eti".

    That default is a convenience for a direct caller, not a licence for an engine:
    an engine that omits it pins its monthly tables to ETI while the rest of its
    reporting follows `context.reporting.interval_kind`. The bivariate and trivariate
    engines had omitted it. Checked as a rule on the source rather than by running a
    fit, so it costs nothing and covers the engines no CI job samples.
    """
    import ast

    tree = ast.parse(inspect.getsource(module))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "emit_monthly_summary"
    ]
    assert calls, f"{module.__name__} does not call emit_monthly_summary"
    for call in calls:
        passed = {kw.arg for kw in call.keywords if kw.arg}
        assert "interval_kind" in passed, (
            f"{module.__name__}:{call.lineno} calls emit_monthly_summary without "
            "interval_kind, so it silently takes the \"eti\" default"
        )

