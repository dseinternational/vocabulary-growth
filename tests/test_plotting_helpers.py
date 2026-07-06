# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

import os
import types

import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from vocab_growth.plotting import (
    _maybe_savgol,
    _resolve_savgol_window_length,
    plot_comprehension_production_gap,
    plot_posterior_kappa,
    plot_posterior_predictive_pmf,
    plot_production_rate,
)


def _ratio_gap_samples(n_plot: int = 20, n_draws: int = 50):
    """Duck-typed stand-in for Bivariate/Trivariate model samples."""
    rng = np.random.default_rng(0)
    return types.SimpleNamespace(
        X_plot=np.linspace(8.0, 36.0, n_plot),
        q_plot=rng.uniform(0.0, 1.0, size=(n_plot, n_draws)),
        p_u_plot=rng.uniform(0.3, 0.6, size=(n_plot, n_draws)),
        p_s_plot=rng.uniform(0.0, 0.3, size=(n_plot, n_draws)),
    )


def test_plot_production_rate_returns_figure_and_writes_files(tmp_path):
    fig = plot_production_rate(
        _ratio_gap_samples(), output_dir=str(tmp_path), filename="prod"
    )
    assert isinstance(fig, Figure)
    for ext in ("png", "svg", "csv"):
        assert os.path.exists(os.path.join(str(tmp_path), f"prod.{ext}"))


def test_plot_comprehension_production_gap_returns_figure_and_writes_files(tmp_path):
    fig = plot_comprehension_production_gap(
        _ratio_gap_samples(), n_trials=800, output_dir=str(tmp_path), filename="gap"
    )
    assert isinstance(fig, Figure)
    for ext in ("png", "svg", "csv"):
        assert os.path.exists(os.path.join(str(tmp_path), f"gap.{ext}"))


def test_plot_posterior_predictive_pmf_does_not_fold_tails_into_endpoints(tmp_path):
    draws = np.array([0] * 5 + [1] * 495 + [2] * 495 + [10] * 5)
    plot_posterior_predictive_pmf(
        X_query=np.array([12.0]),
        X_plot=np.array([12.0]),
        y_plot=draws.reshape(1, -1),
        n_trials=10,
        output_dir=str(tmp_path),
        filename="pmf",
    )

    pmf = pd.read_csv(tmp_path / "pmf.csv")
    assert pmf["word_count"].tolist() == [1, 2]
    np.testing.assert_allclose(pmf["pmf_12m"].to_numpy(), [0.495, 0.495])


def test_plot_posterior_kappa_reports_hdi_not_equal_tailed_interval():
    kappa = np.array([[1.0, 2.0, 3.0, 4.0, 100.0]])
    _fig, df_plot, df_query = plot_posterior_kappa(
        X_plot=np.array([12.0]),
        kappa_plot=kappa,
        X_query=np.array([12.0]),
        kappa_query=kappa,
        n_trials=800,
        hdi_prob=0.60,
    )

    # The 60% HDI excludes the far outlier; the equal-tailed 60% upper bound would
    # be much larger for this deliberately skewed sample.
    assert df_plot.loc[0, "kappa_hi"] < 10.0
    assert df_query.loc[0, "kappa_hi"] < 10.0


@pytest.mark.parametrize("n", [5, 7, 15, 21, 100])
def test_resolve_window_is_valid(n):
    polyorder = 3
    wl = _resolve_savgol_window_length(n, window_length=None, polyorder=polyorder)
    assert wl % 2 == 1          # odd
    assert wl <= n              # fits the data
    assert wl > polyorder       # valid for savgol


def test_resolve_window_raises_when_polyorder_too_high_for_n():
    # n=4 cannot accommodate a cubic (needs an odd window > 3, i.e. >= 5 > n).
    with pytest.raises(ValueError):
        _resolve_savgol_window_length(4, window_length=None, polyorder=3)


def test_resolve_window_even_is_made_odd():
    wl = _resolve_savgol_window_length(50, window_length=20, polyorder=3)
    assert wl % 2 == 1
    assert wl <= 20


def test_resolve_window_below_polyorder_is_bumped():
    # Requesting a window <= polyorder must be raised to a valid odd value.
    wl = _resolve_savgol_window_length(50, window_length=2, polyorder=3)
    assert wl > 3
    assert wl % 2 == 1


def test_resolve_window_too_few_points_raises():
    with pytest.raises(ValueError):
        _resolve_savgol_window_length(2, window_length=None, polyorder=3)


def test_maybe_savgol_passthrough_when_disabled():
    y = np.array([1.0, 5.0, 2.0, 8.0, 3.0])
    out = _maybe_savgol(y, smooth=False, window_length=None, polyorder=2)
    np.testing.assert_array_equal(out, y)
    # The returned array is always float (the smoothing path returns floats too).
    assert out.dtype == float
