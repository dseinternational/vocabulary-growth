# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later


import arviz as az
import dse_research_utils.plot.styles as plot_styles
import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_prior_predictive_prob_curves(
    p_plot_samples,
    x_plot,
    x_obs: pd.Series,
    y_obs: pd.Series,
    x_label: str,
    y_label: str,
    obs_label: str = "Observed",
    y_scale: int = 1,
    n_curves: int = 1000,
    ax: matplotlib.axes.Axes | None = None,
    figsize: tuple[float, float] = plot_styles.FIGSIZE_XL,
):
    """
    Plot sample prior mean-trajectory curves overlaid on observed data.

    Parameters
    ----------
    p_plot_samples : xarray.DataArray
        Prior predictive samples for the mean trajectory.
    x_obs : pd.Series
        Independent variable (e.g., age).
    y_obs : pd.Series
        Dependent variable (e.g., words spoken).
    y_scale : int
        Scale factor for y-axis (e.g., to convert P(x) to count or percentage).
    n_curves : int
    ax : matplotlib.axes.Axes or None

    Returns
    -------
    matplotlib.figure.Figure
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    n_samples = p_plot_samples.shape[1]

    idx = np.random.randint(0, n_samples, n_curves)

    for i in idx:
        ax.plot(
            x_plot,
            p_plot_samples.values[:, i] * y_scale,
            c="#f95800",
            alpha=0.1,
            lw=1,
        )

    ax.scatter(
        x_obs,
        y_obs,
        c="#0058d0",
        alpha=0.4,
        label=obs_label,
    )

    ax.set_xlim(x_plot.min() - 1, x_plot.max() + 1)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    return fig


def plot_prior_predictive_points(
    prior_samples,
    analysis_df: pd.DataFrame,
    n_trials: int,
    n_draws: int = 500,
    ax: matplotlib.axes.Axes | None = None,
):
    """
    Scatter prior predictive observations overlaid on actual data.

    Returns
    -------
    matplotlib.figure.Figure
    """
    y_pred = prior_samples.prior_predictive["y_obs"].stack(sample=("chain", "draw"))
    obs_ages = prior_samples.constant_data["X_obs"].values

    if ax is None:
        fig, ax = plt.subplots(figsize=plot_styles.FIGSIZE_XL)
    else:
        fig = ax.get_figure()

    for i in np.random.randint(0, y_pred.shape[1], n_draws):
        ax.scatter(obs_ages, y_pred.values[:, i], color="coral", alpha=0.01, s=12)

    ax.scatter(
        analysis_df["age"],
        analysis_df["spoken"],
        alpha=0.43,
        s=12,
        label="Actual Data",
    )
    ax.set_xlabel("Age (months)")
    ax.set_ylabel("Words spoken")
    ax.set_ylim(-20, n_trials + 50)

    return fig


def plot_prior_predictive_ppc(
    prior_samples,
    random_seed: int = 47,
):
    """
    ArviZ prior predictive check plot.

    Returns
    -------
    matplotlib.figure.Figure
    """
    az.plot_ppc(
        prior_samples,
        group="prior",
        observed=True,
        random_seed=random_seed,
        figsize=plot_styles.FIGSIZE_MD,
    )
    fig = plt.gcf()
    return fig
