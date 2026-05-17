# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

import os

import arviz as az
import dse_research_utils.plot.predictive as plot_predictive
import dse_research_utils.plot.styles as plot_styles
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.figure import Figure
from scipy.signal import savgol_filter


def _save_csv(df: pd.DataFrame, output_dir: str, filename: str) -> None:
    """Save a DataFrame as CSV alongside the corresponding plot."""
    df.to_csv(os.path.join(output_dir, f"{filename}.csv"), index=False)


def _hdi_by_sample(values: np.ndarray, prob: float) -> np.ndarray:
    """Compute HDI over posterior samples for each plot point."""
    values_da = xr.DataArray(values, dims=("sample", "plot"))
    return az.hdi(values_da, prob=prob, dim="sample").to_numpy()


def plot_eta_effect_sizes(
    eta_values=None,
    p_values=None,
    n_trials=800,
    output_dir=None,
    filename=None):
    """
    Show how the amplitude η translates to word-count effects.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if eta_values is None:
        eta_values = [0.3, 0.5, 0.7, 1.0, 1.5]
    if p_values is None:
        p_values = np.linspace(0.01, 0.99, 100)
    fig, ax = plt.subplots(figsize=plot_styles.FIGSIZE_MD)

    for eta in eta_values:
        effects = n_trials * p_values * (1 - p_values) * eta
        word_counts = p_values * n_trials
        ax.plot(word_counts, effects, label=f"η = {eta}", lw=2)

    ax.set_xlabel("Vocabulary size (words)")
    ax.set_ylabel("Effect size (±words for 1 SD of GP)")
    ax.legend()

    if output_dir is not None and filename is not None:
        fig.savefig(os.path.join(output_dir, f"{filename}.png"), dpi=300)
        fig.savefig(os.path.join(output_dir, f"{filename}.svg"))
        csv_data = {"vocabulary_size": p_values * n_trials}
        for eta in eta_values:
            csv_data[f"effect_eta_{eta}"] = n_trials * p_values * (1 - p_values) * eta
        _save_csv(pd.DataFrame(csv_data), output_dir, filename)

    return fig

# ------------------------------------------------------------
# Prior predictive plots
# ------------------------------------------------------------

def plot_prior_samples(
    x: np.ndarray,
    y_samples: np.ndarray,
    x_obs: np.ndarray | pd.Series,
    y_obs: np.ndarray | pd.Series,
    n_trials: int = 800,
    n_curves = 1000,
    x_label: str = "x",
    y_label: str = "y",
    filename: str | None = None,
    output_dir: str | None = None,
) -> Figure:

    fig = plot_predictive.plot_prior_samples_binomial(
        x,
        y_samples,
        x_obs,
        y_obs,
        n_trials,
        n_curves,
        x_label,
        y_label,
        filename,
        output_dir)

    return fig


def plot_prior_predictions(
    x: np.ndarray,
    y_pred: np.ndarray,
    x_obs: np.ndarray | pd.Series,
    y_obs: np.ndarray | pd.Series,
    n_trials: int = 800,
    x_label: str = "x",
    y_label: str = "y",
    filename: str | None = None,
    output_dir: str | None = None,
) -> Figure:

    plt.figure(figsize=plot_styles.FIGSIZE_XL)

    for i in np.random.randint(0, y_pred.shape[1], 500):
        plt.scatter(x, y_pred[:, i], color=plot_styles.COLOUR_ORANGE, alpha=0.01, s=12)

    plt.scatter(
        x_obs,
        y_obs,
        color=plot_styles.COLOUR_BLUE,
        alpha=0.4,
        s=10,
        label="Observed data",
    )

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.ylim(-20, n_trials + 50)

    if filename is not None and output_dir is not None:
        plt.savefig(os.path.join(output_dir, f"{filename}.png"), dpi=300)

    return plt.gcf()


def plot_posterior_predictive_count_distributions_by_query_age(
    X_query: np.ndarray,
    y_query: np.ndarray,
    n_trials: int,
    bin_width: int = 5,
    plot_cols: int = 2,
    hdi_prob: float = 0.89,
    eti_prob: float | None = None,
    output_dir: str | None = None,
    filename: str | None = None,
    x_label: str = "Word count",
) -> Figure:
    """
    For each query age, plot the posterior predictive distribution of counts, as a histogram.
    """
    bins = np.arange(0, n_trials + bin_width + 1, bin_width)
    centres = (bins[:-1] + bins[1:]) / 2
    nq = len(X_query)

    plot_rows = int(np.ceil(nq / plot_cols))
    fig, axes = plt.subplots(plot_rows, plot_cols, figsize=(10, 3.8 * plot_rows), sharex=False)
    axes = np.atleast_1d(axes).ravel()

    # ETI quantile levels matching context.reporting.hdi mass (e.g., 0.90 -> [0.05, 0.95])
    if eti_prob is not None:
        q_lo = (1.0 - eti_prob) / 2.0
        q_hi = 1.0 - q_lo
    else:
        q_lo, q_hi = 0, 0

    for j, age in enumerate(X_query):
        ax = axes[j]
        draws = y_query[j, :].astype(int)
        counts, _ = np.histogram(draws, bins=bins)
        pmf_bins = counts / counts.sum()

        med = np.median(draws)
        hdi = az.hdi(draws, prob=hdi_prob)  # (lo, hi)

        if eti_prob is not None:
            eti_lo, eti_hi = np.quantile(draws, [q_lo, q_hi])
        else:
            eti_lo, eti_hi = 0, 0

        ylim_max = max(pmf_bins.max() * 1.08, 0.25)

        # HDI
        ax.fill_betweenx(
            [0, ylim_max * 0.96], hdi[0], hdi[1], color=plot_styles.COLOUR_GREEN, alpha=0.10
        )
        ax.text(
            hdi[1] + 150,
            ylim_max * 0.9,
            f"{int(hdi_prob*100)}% HPDI: {hdi[0]:.0f} to {hdi[1]:.0f}",
            color=plot_styles.COLOUR_GREEN,
            ha="center",
        )

        if eti_prob is not None:
            ax.fill_betweenx([0, ylim_max * 0.96], eti_lo, eti_hi, color=plot_styles.COLOUR_ORANGE, alpha=0.10)
            ax.text(
                eti_hi + 150,
                ylim_max * 0.82,
                f"{int(eti_prob*100)}% ETI: {eti_lo:.0f} to {eti_hi:.0f}",
                color=plot_styles.COLOUR_ORANGE,
                ha="center",
            )

        # PMF histogram
        ax.bar(centres, pmf_bins, width=bin_width, color=plot_styles.COLOUR_BLUE, align="center")

        # median
        ax.axvline(med, lw=2, ls="--", color=plot_styles.COLOUR_RED)
        ax.text(
            med + 110,
            ylim_max * 0.98,
            f"median: {med:.0f}",
            color=plot_styles.COLOUR_RED,
            ha="center",
        )

        ax.set_title(f"{age:.1f} months")
        ax.set_ylim(0, ylim_max)
        ax.set_ylabel(f"Posterior predictive probability mass (bin width {bin_width})")
        ax.set_xlim(0, n_trials)
        ax.set_xticks(np.arange(0, n_trials + 1, 100))

    for k in range(nq, len(axes)):
        axes[k].axis("off")

    for ax in axes[max(0, len(axes) - plot_cols) :]:
        ax.set_xlabel(f"{x_label} (bins of {bin_width})")

    fig.suptitle("Posterior predictive distributions at query ages", y=1.02)

    if filename is not None and output_dir is not None:
        plt.savefig(os.path.join(output_dir, f"{filename}.png"), dpi=300)
        plt.savefig(os.path.join(output_dir, f"{filename}.svg"))
        rows = []
        for j, age in enumerate(X_query):
            draws = y_query[j, :].astype(int)
            med = np.median(draws)
            hdi_bounds = az.hdi(draws, prob=hdi_prob)
            row = {"age_months": age, "median": med, "hdi_lo": hdi_bounds[0], "hdi_hi": hdi_bounds[1]}
            if eti_prob is not None:
                row["eti_lo"], row["eti_hi"] = np.quantile(draws, [q_lo, q_hi])
            rows.append(row)
        _save_csv(pd.DataFrame(rows), output_dir, filename)

    return fig

def plot_posterior_predictive_pmf(
    X_query: np.ndarray,
    X_plot: np.ndarray,
    y_plot: np.ndarray,
    n_trials: int,
    log_scale: bool = False,
    output_dir: str | None = None,
    filename: str | None = None,
    x_label: str = "Word count",
) -> Figure:
    """
    For each query age, plot the posterior predictive distribution of counts as a PMF on a common support.
    """
    all_draws = []
    idxs = []
    for a in X_query:
        j = int(np.argmin(np.abs(X_plot - a)))
        idxs.append(j)
        all_draws.append(y_plot[j, :].astype(int))
    all_draws = np.concatenate(all_draws)
    x_lo, x_hi = np.quantile(all_draws, [0.01, 0.99])
    x_lo = int(max(0, np.floor(x_lo)))
    x_hi = int(min(n_trials, np.ceil(x_hi)))

    k = np.arange(x_lo, x_hi + 1)
    plt.figure(figsize=plot_styles.FIGSIZE_XL)

    for _a, j in zip(X_query, idxs, strict=True):
        draws = np.clip(y_plot[j, :].astype(int), x_lo, x_hi)

        # Empirical PMF on common support
        counts = np.bincount(draws - x_lo, minlength=len(k))
        pmf = counts[: len(k)] / counts.sum()
        # Step line (discrete PMF)
        plt.step(k, pmf, where="mid", lw=2, label=f"{X_plot[j]:.0f}m")

    plt.xlabel(x_label)
    plt.ylabel("Posterior predictive probability")
    plt.title("Posterior predictive PMF at selected ages")
    plt.xlim(x_lo, x_hi)

    if log_scale:
        plt.yscale("log")

    plt.legend(title="Age", ncol=2, frameon=True)

    if filename is not None and output_dir is not None:
        plt.savefig(os.path.join(output_dir, f"{filename}.png"), dpi=300)
        plt.savefig(os.path.join(output_dir, f"{filename}.svg"))
        csv_data = {"word_count": k}
        for _a, j in zip(X_query, idxs, strict=True):
            draws = np.clip(y_plot[j, :].astype(int), x_lo, x_hi)
            counts = np.bincount(draws - x_lo, minlength=len(k))
            pmf = counts[: len(k)] / counts.sum()
            csv_data[f"pmf_{X_plot[j]:.0f}m"] = pmf
        _save_csv(pd.DataFrame(csv_data), output_dir, filename)

    return plt.gcf()



def plot_posterior_predictive_cdf(
    X_query: np.ndarray,
    X_plot: np.ndarray,
    y_plot: np.ndarray,
    n_trials: int,
    output_dir: str | None = None,
    filename: str | None = None,
    x_label: str = "Words spoken (count)",
) -> Figure:
    draws_by_age = []
    plot_idx_by_age = []

    for a in X_query:
        j = int(np.argmin(np.abs(X_plot - a)))
        plot_idx_by_age.append(j)
        draws_by_age.append(y_plot[j, :].astype(int))

    # Choose a common x-range so curves are comparable (central 99% over all selected ages)
    all_draws = np.concatenate(draws_by_age)
    x_lo, x_hi = np.quantile(all_draws, [0.005, 0.995])
    x_lo = int(max(0, np.floor(x_lo)))
    x_hi = int(min(n_trials, np.ceil(x_hi)))
    k = np.arange(x_lo, x_hi + 1)

    plt.figure(figsize=plot_styles.FIGSIZE_XL)

    for _a, j, draws in zip(X_query, plot_idx_by_age, draws_by_age, strict=True):
        # Empirical CDF on common support: F(k) = mean(draws <= k)
        # Vectorised computation:
        draws_sorted = np.sort(draws)
        cdf = np.searchsorted(draws_sorted, k, side="right") / draws_sorted.size
        plt.step(k, cdf, where="post", lw=2, label=f"{X_plot[j]:.0f}m")

    plt.xlabel(x_label)
    plt.ylabel("Posterior predictive CDF  P(Y ≤ k)")
    plt.title("Posterior predictive CDFs at selected ages")
    plt.xlim(x_lo, x_hi)
    plt.ylim(0, 1)
    plt.legend(title="Age", ncol=2, frameon=True)

    if filename is not None and output_dir is not None:
        plt.savefig(os.path.join(output_dir, f"{filename}.png"), dpi=300)
        plt.savefig(os.path.join(output_dir, f"{filename}.svg"))
        csv_data = {"word_count": k}
        for _a, j, draws in zip(X_query, plot_idx_by_age, draws_by_age, strict=True):
            draws_sorted = np.sort(draws)
            cdf = np.searchsorted(draws_sorted, k, side="right") / draws_sorted.size
            csv_data[f"cdf_{X_plot[j]:.0f}m"] = cdf
        _save_csv(pd.DataFrame(csv_data), output_dir, filename)

    return plt.gcf()


def _resolve_savgol_window_length(
    n: int,
    window_length: int | None,
    polyorder: int,
) -> int:
    """
    Resolve a valid Savitzky-Golay window length.

    Ensures:
    - odd integer
    - <= n
    - > polyorder
    """
    if n < 3:
        raise ValueError("Need at least 3 points to apply Savitzky-Golay smoothing.")

    if window_length is None:
        # Sensible default: modest smoothing, scale with grid size
        window_length = min(21, n)

    if window_length < 3:
        window_length = 3

    if window_length > n:
        window_length = n

    # Must be odd
    if window_length % 2 == 0:
        window_length -= 1

    # Must be greater than polyorder
    min_valid = polyorder + 2 if (polyorder + 2) % 2 == 1 else polyorder + 3
    if window_length <= polyorder:
        window_length = min_valid

    if window_length > n:
        raise ValueError(
            f"Cannot choose a valid Savitzky-Golay window_length for n={n} and polyorder={polyorder}."
        )

    if window_length % 2 == 0:
        window_length -= 1

    if window_length <= polyorder:
        raise ValueError(
            f"window_length must be greater than polyorder. Got window_length={window_length}, polyorder={polyorder}."
        )

    return window_length


def _maybe_savgol(
    y: np.ndarray,
    smooth: bool,
    window_length: int | None,
    polyorder: int,
) -> np.ndarray:
    """
    Optionally apply Savitzky-Golay smoothing to a 1D array.
    """
    y = np.asarray(y, dtype=float)

    if not smooth:
        return y

    wl = _resolve_savgol_window_length(
        n=len(y),
        window_length=window_length,
        polyorder=polyorder,
    )

    return savgol_filter(y, window_length=wl, polyorder=polyorder)


def plot_posterior_predictive_median_trend(
    X_plot: np.ndarray,
    y_plot: np.ndarray,
    x_obs: np.ndarray | pd.Series,
    y_obs: np.ndarray | pd.Series,
    smooth: bool = False,
    savgol_window_length: int | None = None,
    savgol_polyorder: int = 3,
    smooth_intervals: bool = True,
    output_dir: str | None = None,
    filename: str | None = None,
    y_label: str = "Predicted word count",
):
    """
    Plot the posterior predictive distribution of counts as a function of age,
    showing the predictive median and multiple predictive percentile intervals.

    Parameters
    ----------
    X_plot
        Age grid, shape (n_grid,) or (n_grid, 1).
    y_plot
        Posterior predictive samples, shape (n_grid, n_samples).
    x_obs
        Observed ages.
    y_obs
        Observed counts.
    output_dir
        Output directory for saved figure.
    filename
        Base filename for saved figure.
    smooth
        If True, apply Savitzky-Golay smoothing to plotted summaries.
    savgol_window_length
        Window length for Savitzky-Golay smoothing. Must be odd and > polyorder.
        If None, a default is chosen.
    savgol_polyorder
        Polynomial order for Savitzky-Golay smoothing.
    smooth_intervals
        If True, smooth interval bounds as well as the median curve.
        If False, smooth only the median.
    y_label
        Label for the y-axis.
    """
    X_plot = np.asarray(X_plot).reshape(-1)
    y_plot = np.asarray(y_plot)
    x_obs = np.asarray(x_obs).reshape(-1)
    y_obs = np.asarray(y_obs).reshape(-1)

    if y_plot.ndim != 2:
        raise ValueError("y_plot must have shape (n_grid, n_samples).")

    if len(X_plot) != y_plot.shape[0]:
        raise ValueError(
            f"X_plot length ({len(X_plot)}) must match y_plot.shape[0] ({y_plot.shape[0]})."
        )

    y_plot_samples_median = np.quantile(y_plot, 0.50, axis=1)

    predictive_interval_1 = np.quantile(y_plot, [0.05, 0.95], axis=1).T
    predictive_interval_2 = np.quantile(y_plot, [0.25, 0.75], axis=1).T
    predictive_interval_3 = np.quantile(y_plot, [0.375, 0.625], axis=1).T

    # Optional smoothing for display
    y_plot_samples_median_plot = _maybe_savgol(
        y_plot_samples_median,
        smooth=smooth,
        window_length=savgol_window_length,
        polyorder=savgol_polyorder,
    )

    if smooth and smooth_intervals:
        predictive_interval_1_plot = np.column_stack(
            [
                _maybe_savgol(
                    predictive_interval_1[:, 0],
                    smooth=True,
                    window_length=savgol_window_length,
                    polyorder=savgol_polyorder,
                ),
                _maybe_savgol(
                    predictive_interval_1[:, 1],
                    smooth=True,
                    window_length=savgol_window_length,
                    polyorder=savgol_polyorder,
                ),
            ]
        )
        predictive_interval_2_plot = np.column_stack(
            [
                _maybe_savgol(
                    predictive_interval_2[:, 0],
                    smooth=True,
                    window_length=savgol_window_length,
                    polyorder=savgol_polyorder,
                ),
                _maybe_savgol(
                    predictive_interval_2[:, 1],
                    smooth=True,
                    window_length=savgol_window_length,
                    polyorder=savgol_polyorder,
                ),
            ]
        )
        predictive_interval_3_plot = np.column_stack(
            [
                _maybe_savgol(
                    predictive_interval_3[:, 0],
                    smooth=True,
                    window_length=savgol_window_length,
                    polyorder=savgol_polyorder,
                ),
                _maybe_savgol(
                    predictive_interval_3[:, 1],
                    smooth=True,
                    window_length=savgol_window_length,
                    polyorder=savgol_polyorder,
                ),
            ]
        )
    else:
        predictive_interval_1_plot = predictive_interval_1
        predictive_interval_2_plot = predictive_interval_2
        predictive_interval_3_plot = predictive_interval_3

    plt.figure(figsize=plot_styles.FIGSIZE_XL)

    plt.fill_between(
        X_plot,
        predictive_interval_1_plot[:, 0],
        predictive_interval_1_plot[:, 1],
        alpha=0.20,
        label="90% predictive percentile interval (equal-tailed)",
    )
    plt.fill_between(
        X_plot,
        predictive_interval_2_plot[:, 0],
        predictive_interval_2_plot[:, 1],
        alpha=0.30,
        label="50% predictive percentile interval (equal-tailed)",
    )
    plt.fill_between(
        X_plot,
        predictive_interval_3_plot[:, 0],
        predictive_interval_3_plot[:, 1],
        alpha=0.40,
        label="25% predictive percentile interval (equal-tailed)",
    )
    plt.plot(
        X_plot,
        y_plot_samples_median_plot,
        lw=3,
        label="Posterior median (predictive)",
    )
    plt.scatter(
        x_obs,
        y_obs,
        s=15,
        alpha=0.25,
        label="Observed",
    )

    plt.xlabel("Age (months)")
    plt.ylabel(y_label)
    plt.legend(loc="upper left", frameon=True)
    plt.ylim(-20, np.max(y_plot) + 50)

    if filename is not None and output_dir is not None:
        plt.savefig(os.path.join(output_dir, f"{filename}.png"), dpi=300)
        plt.savefig(os.path.join(output_dir, f"{filename}.svg"))
        _save_csv(pd.DataFrame({
            "age_months": X_plot,
            "median": y_plot_samples_median,
            "p05": predictive_interval_1[:, 0],
            "p95": predictive_interval_1[:, 1],
            "p25": predictive_interval_2[:, 0],
            "p75": predictive_interval_2[:, 1],
            "p375": predictive_interval_3[:, 0],
            "p625": predictive_interval_3[:, 1],
        }), output_dir, filename)

    return plt.gcf()


def plot_expected_learning_rate(
    X_plot: np.ndarray,
    f_plot: np.ndarray,
    n_trials: int,
    hdi_prob: float = 0.90,
    output_dir: str | None = None,
    filename: str | None = None,
    smooth: bool = False,
    savgol_window_length: int | None = None,
    savgol_polyorder: int = 3,
    smooth_intervals: bool = True,
    y_label: str = "Estimated word score gain per month",
):
    """
    Plot the posterior distribution of the estimated learning rate
    (estimated gain in spoken words per month) across age.

    The is the derivative of the conditional expectation of the count given the
    latent function f. The posterior uncertainty bands show how that estimated
    rate varies across posterior draws of f:

        E[Y] = n_trials * sigmoid(f)
        dE[Y]/dx = n_trials * p * (1 - p) * df/dx

    Parameters
    ----------
    X_plot
        Age grid in months, shape ``(n_plot,)``.
    f_plot
        Posterior samples of the latent linear predictor at the plot grid,
        shape ``(n_plot, n_samples)``.
    n_trials
        Maximum count / checklist size.
    hdi_prob
        Probability mass for the outer HDI band.
    output_dir
        Directory to save figure files.
    filename
        Base filename (without extension) for saved figure.
    smooth
        If True, apply Savitzky-Golay smoothing to plotted summaries.
    savgol_window_length
        Window length for Savitzky-Golay smoothing. Must be odd and > polyorder.
        If None, a default is chosen.
    savgol_polyorder
        Polynomial order for Savitzky-Golay smoothing.
    smooth_intervals
        If True, smooth HDI bounds as well as the median curve.
        If False, smooth only the median.
    y_label
        Label for the y-axis.

    Returns
    -------
    matplotlib.figure.Figure
    """
    x_plot_values = np.asarray(X_plot, dtype=float).reshape(-1)
    # f_plot arrives as (n_plot, n_samples); transpose to (n_samples, n_plot)
    f_plot_values = np.asarray(f_plot).T

    if f_plot_values.shape[1] != x_plot_values.shape[0]:
        raise ValueError(
            f"Shape mismatch: f_plot has {f_plot_values.shape[1]} plot points, "
            f"but X_plot has {x_plot_values.shape[0]}."
        )

    if x_plot_values.shape[0] < 3:
        raise ValueError("At least 3 plot points are required to compute a gradient.")

    # Derivative df/dx along age axis
    dfdx = np.gradient(
        f_plot_values,
        x_plot_values,
        axis=1,
        edge_order=2,
    )  # shape: (n_samples, n_plot)

    # Transform to estimated word-count learning rate
    #    E[Y] = N * sigmoid(f)
    #    dE[Y]/dx = N * p * (1 - p) * df/dx
    p = 1.0 / (1.0 + np.exp(-f_plot_values))
    rate = n_trials * p * (1.0 - p) * dfdx  # words per month

    # Posterior summaries across draws, per age point
    median_rate = np.median(rate, axis=0)
    hdi_rate = _hdi_by_sample(rate, prob=hdi_prob)
    hdi_75_rate = _hdi_by_sample(rate, prob=0.75)
    hdi_50_rate = _hdi_by_sample(rate, prob=0.50)

    # Optional smoothing for display
    median_rate_plot = _maybe_savgol(
        median_rate,
        smooth=smooth,
        window_length=savgol_window_length,
        polyorder=savgol_polyorder,
    )

    if smooth and smooth_intervals:
        hdi_rate_plot = np.column_stack(
            [
                _maybe_savgol(
                    hdi_rate[:, 0],
                    smooth=True,
                    window_length=savgol_window_length,
                    polyorder=savgol_polyorder,
                ),
                _maybe_savgol(
                    hdi_rate[:, 1],
                    smooth=True,
                    window_length=savgol_window_length,
                    polyorder=savgol_polyorder,
                ),
            ]
        )
        hdi_75_rate_plot = np.column_stack(
            [
                _maybe_savgol(
                    hdi_75_rate[:, 0],
                    smooth=True,
                    window_length=savgol_window_length,
                    polyorder=savgol_polyorder,
                ),
                _maybe_savgol(
                    hdi_75_rate[:, 1],
                    smooth=True,
                    window_length=savgol_window_length,
                    polyorder=savgol_polyorder,
                ),
            ]
        )
        hdi_50_rate_plot = np.column_stack(
            [
                _maybe_savgol(
                    hdi_50_rate[:, 0],
                    smooth=True,
                    window_length=savgol_window_length,
                    polyorder=savgol_polyorder,
                ),
                _maybe_savgol(
                    hdi_50_rate[:, 1],
                    smooth=True,
                    window_length=savgol_window_length,
                    polyorder=savgol_polyorder,
                ),
            ]
        )
    else:
        hdi_rate_plot = hdi_rate
        hdi_75_rate_plot = hdi_75_rate
        hdi_50_rate_plot = hdi_50_rate

    plt.figure(figsize=plot_styles.FIGSIZE_XL)

    plt.fill_between(
        x_plot_values,
        hdi_rate_plot[:, 0],
        hdi_rate_plot[:, 1],
        alpha=0.20,
        label=f"{int(hdi_prob * 100)}% HDI",
    )
    plt.fill_between(
        x_plot_values,
        hdi_75_rate_plot[:, 0],
        hdi_75_rate_plot[:, 1],
        alpha=0.25,
        label="75% HDI",
    )
    plt.fill_between(
        x_plot_values,
        hdi_50_rate_plot[:, 0],
        hdi_50_rate_plot[:, 1],
        alpha=0.30,
        label="50% HDI",
    )
    plt.plot(
        x_plot_values,
        median_rate_plot,
        lw=3,
        label="Median estimated gain per month",
    )

    plt.xlabel("Age (months)")
    plt.ylabel(y_label)
    plt.legend(loc="upper left", frameon=True)

    if filename is not None and output_dir is not None:
        plt.savefig(os.path.join(output_dir, f"{filename}.png"), dpi=300)
        plt.savefig(os.path.join(output_dir, f"{filename}.svg"))
        _save_csv(pd.DataFrame({
            "age_months": x_plot_values,
            "median_rate": median_rate,
            "hdi_lo": hdi_rate[:, 0],
            "hdi_hi": hdi_rate[:, 1],
            "hdi75_lo": hdi_75_rate[:, 0],
            "hdi75_hi": hdi_75_rate[:, 1],
            "hdi50_lo": hdi_50_rate[:, 0],
            "hdi50_hi": hdi_50_rate[:, 1],
        }), output_dir, filename)

    return plt.gcf()


def plot_posterior_kappa(
    X_plot: np.ndarray,
    kappa_plot: np.ndarray,
    X_query: np.ndarray,
    kappa_query: np.ndarray,
    n_trials: int,
    hdi_prob: float = 0.90,
    output_dir: str | None = None,
    filename: str | None = None,
) -> tuple[Figure, pd.DataFrame, pd.DataFrame]:
    """
    Plot the posterior distribution of κ(age) on the plot grid, and return
    summary DataFrames for both the plot grid and query ages.

    Parameters
    ----------
    X_plot
        Age grid in months for the plot points, shape ``(n_plot,)``.
    kappa_plot
        Posterior samples of κ at the plot grid,
        shape ``(n_plot, n_samples)``.
    X_query
        Age grid in months for the query points, shape ``(n_query,)``.
    kappa_query
        Posterior samples of κ at the query ages,
        shape ``(n_query, n_samples)``.
    n_trials
        Maximum count / checklist size.
    hdi_prob
        Probability mass for the credible band.
    output_dir
        Directory to save figure files.
    filename
        Base filename (without extension) for saved figure.

    Returns
    -------
    tuple[Figure, pd.DataFrame, pd.DataFrame]
        ``(fig, df_kappa_plot, df_kappa_query)`` where each DataFrame contains
        columns for age, median κ, credible interval bounds, intra-cluster
        correlation ρ, and variance inflation factor.
    """
    X_plot = np.asarray(X_plot, dtype=float).reshape(-1)
    kappa_plot_samps = np.asarray(kappa_plot)
    X_query = np.asarray(X_query, dtype=float).reshape(-1)
    kappa_query_samps = np.asarray(kappa_query)

    q_lo = (1.0 - hdi_prob) / 2.0
    q_hi = 1.0 - q_lo

    # --- Plot grid ---
    kappa_plot_lo = np.quantile(kappa_plot_samps, q_lo, axis=1)
    kappa_plot_med = np.quantile(kappa_plot_samps, 0.5, axis=1)
    kappa_plot_hi = np.quantile(kappa_plot_samps, q_hi, axis=1)

    df_kappa_plot = pd.DataFrame(
        {
            "age_months": X_plot,
            "kappa_lo": kappa_plot_lo,
            "kappa_median": kappa_plot_med,
            "kappa_hi": kappa_plot_hi,
            "rho_median": 1.0 / (kappa_plot_med + 1.0),
            "vif_median": (n_trials + kappa_plot_med) / (1.0 + kappa_plot_med),
        }
    )

    # --- Query ages ---

    kappa_query_lo = np.quantile(kappa_query_samps, q_lo, axis=1)
    kappa_query_med = np.quantile(kappa_query_samps, 0.5, axis=1)
    kappa_query_hi = np.quantile(kappa_query_samps, q_hi, axis=1)

    df_kappa_query = pd.DataFrame(
        {
            "age_months": X_query,
            "kappa_lo": kappa_query_lo,
            "kappa_median": kappa_query_med,
            "kappa_hi": kappa_query_hi,
            "rho_median": np.median(1.0 / (kappa_query_samps + 1.0), axis=1),
            "vif_median": np.median(
                (n_trials + kappa_query_samps) / (1.0 + kappa_query_samps), axis=1
            ),
        }
    )

    # --- Plot ---
    fig, ax = plt.subplots(figsize=plot_styles.FIGSIZE_XL)

    ax.fill_between(
        X_plot,
        kappa_plot_lo,
        kappa_plot_hi,
        alpha=0.25,
        label=f"{int(hdi_prob * 100)}% credible interval",
    )
    ax.plot(X_plot, kappa_plot_med, lw=2.5, label="Median κ(age)")

    ax.set_yscale("log")
    ax.set_xlabel("Age (months)")
    ax.set_ylabel("κ(age) (log scale)")
    ax.set_title("Posterior κ(age) with credible band")
    ax.legend(frameon=True)

    if filename is not None and output_dir is not None:
        fig.savefig(os.path.join(output_dir, f"{filename}.png"), dpi=300)
        fig.savefig(os.path.join(output_dir, f"{filename}.svg"))
        _save_csv(df_kappa_plot, output_dir, filename)

    return fig, df_kappa_plot, df_kappa_query
