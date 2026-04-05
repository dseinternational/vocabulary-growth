# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Model VG01: Influence of age on words spoken (A → S) - children with Down syndrome
"""

import os
import shutil

import arviz as az
import dse_research_utils.environment.info as env_info
import dse_research_utils.math.constants as math_constants
import dse_research_utils.metadata.packages as package_metadata
import dse_research_utils.plot.diagnostics_mcmc as plot_diagnostics_mcmc
import dse_research_utils.plot.distributions as plot_dist
import dse_research_utils.plot.predictive as plot_predictive
import dse_research_utils.plot.styles as plot_styles
import dse_research_utils.statistics.descriptive as descriptive_stats
import dse_research_utils.statistics.models.data as model_data
import dse_research_utils.statistics.models.pymc_utils as pymc_utils
import dse_research_utils.statistics.models.reporting as reporting
import dse_research_utils.statistics.models.sampling as sampling
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import preliz as pz
import pymc as pm
from arviz import InferenceData
from rich import print
from rich.pretty import pprint

import vocab_growth.data_utils as vocab_data_utils
import vocab_growth.environment as local_env
import vocab_growth.plotting as plotting
import vocab_growth.posterior_analysis as posterior_analysis
from vocab_growth.models.common import ModelConfiguration, ModelFitContext, ModelSamples


def prepare_model_data(
    x_col: str = "age",
    y_col: str = "spoken",
    max_age_months: int | None = None,
    n_trials: int = 800,
) -> tuple[model_data.BinomialModelData, pd.DataFrame, pd.DataFrame]:

    vocab_df = vocab_data_utils.load_combined_data(max_age_months=max_age_months)
    analysis_df = vocab_df[[x_col, y_col]].dropna()

    desc = descriptive_stats.describe_all(analysis_df, alpha=0.05)

    print(
        "\n[green]------------------------------------------------------------[/green]"
    )
    print("[bold green]Descriptive statistics[/bold green]")
    print("[green]------------------------------------------------------------[/green]")

    pprint(desc)

    X_obs = np.asarray(analysis_df[x_col], dtype=float).reshape(-1, 1)  # (n, 1)
    y_obs = np.asarray(analysis_df[y_col], dtype=int)  # (n,)

    return (
        model_data.BinomialModelData(X_obs=X_obs, y_obs=y_obs, n_trials=n_trials),
        analysis_df,
        desc,
    )


def build_model(context: ModelFitContext):
    """
    Builds vocabulary growth model.
    """
    print(
        "\n[green]------------------------------------------------------------[/green]"
    )
    print("[bold green]Model definition and initialisation[/bold green]")
    print("[green]------------------------------------------------------------[/green]")
    print()

    n = len(context.model_data.y_obs)

    if context.model_data.X_obs.shape[0] != n:
        raise ValueError("X_obs and y_obs have inconsistent lengths.")
    if not np.all(0 <= context.model_data.y_obs):
        raise ValueError("y_obs contains negative counts.")
    if not np.all(context.model_data.y_obs <= context.model_data.n_trials):
        raise ValueError("y_obs exceeds n_trials.")

    print(
        "\n[green]------------------------------------------------------------[/green]"
    )
    print("[bold green]Building vocabulary growth model[/bold green]")
    print("[green]------------------------------------------------------------[/green]")

    print(f"  Number of observations: {n}")
    print(f"  Number of trials (n_trials): {context.model_data.n_trials}")
    print(f"  Slope anchors (months): {context.model_config.slope_anchors}")
    print(f"  Length-scale range (months): {context.model_config.ell_months_range}")
    print(f"  Prior for p_slope_low: {context.model_config.p_slope_low_dist}")
    print(f"  Prior for p_slope_hi: {context.model_config.p_slope_hi_dist}")
    print(f"  Prior for ell_unit: {context.model_config.ell_unit_dist}")
    print(f"  Prior for eta: {context.model_config.eta_dist}")
    print(f"  Prior for kappa_min: {context.model_config.kappa_min_dist}")
    print(f"  Prior for a_kappa: {context.model_config.a_kappa_dist}")
    print(f"  Prior for b_kappa_mag: {context.model_config.b_kappa_mag_dist}")
    print(f"  Number of plot points: {context.model_config.n_plot}")

    X_obs_median = float(np.median(context.model_data.X_obs))
    X_obs_mean = float(np.mean(context.model_data.X_obs))
    X_obs_std = float(np.std(context.model_data.X_obs, ddof=1))

    if not np.isfinite(X_obs_std) or X_obs_std <= 0:
        raise ValueError("Age standard deviation must be positive.")

    print(
        f"  Age (months) - median: {X_obs_median:.2f}, mean: {X_obs_mean:.2f}, std: {X_obs_std:.2f}"
    )

    # Predictor standardised (z-score)
    X_obs_z = (context.model_data.X_obs - X_obs_mean) / X_obs_std

    # Plot grid
    X_plot = np.linspace(
        context.model_data.X_obs.min(),
        context.model_data.X_obs.max(),
        context.model_config.n_plot,
    ).reshape(-1, 1)
    X_plot_z = (X_plot - X_obs_mean) / X_obs_std

    # Query grid
    X_query = np.array(context.model_config.ages_query).reshape(-1, 1)
    X_query_z = (X_query - X_obs_mean) / X_obs_std

    # Stack for 'free' predictions (all standardised)
    X_all_z = np.vstack([X_obs_z, X_plot_z, X_query_z])

    n_plot = X_plot_z.shape[0]
    n_query = X_query_z.shape[0]
    n_all = X_all_z.shape[0]

    ell_low_months = float(context.model_config.ell_months_range[0])
    ell_high_months = float(context.model_config.ell_months_range[1])

    if ell_low_months <= 0 or ell_high_months <= 0:
        raise ValueError("Length-scale bounds must be positive (in months).")
    if ell_high_months <= ell_low_months:
        raise ValueError("ell_months_range must be (low, high) with high > low.")

    ell_low_z = ell_low_months / X_obs_std
    ell_high_z = ell_high_months / X_obs_std
    ell_range_z = (ell_low_z, ell_high_z)

    L, M = get_hsgp_hyperparams(
        X_obs_z,
        ell_range_z,
    )

    print(f"  HSGP basis size (m): {M}")
    print(f"  HSGP L: {L}")

    slope_age_a = float(context.model_config.slope_anchors[0])
    slope_age_b = float(context.model_config.slope_anchors[1])
    slope_age_a_z = (slope_age_a - X_obs_mean) / X_obs_std
    slope_age_b_z = (slope_age_b - X_obs_mean) / X_obs_std

    print(f"  Slope anchors (z-scores): {slope_age_a_z:.2f}, {slope_age_b_z:.2f}")

    i_obs0, i_obs1 = 0, X_obs_z.shape[0]
    i_plot0, i_plot1 = i_obs1, i_obs1 + X_plot_z.shape[0]
    i_query0, i_query1 = i_plot1, i_plot1 + X_query_z.shape[0]

    print(
        f"  Query points: {', '.join(map(str, context.model_config.ages_query))} (months)"
    )

    coords = {
        "all_id": np.arange(n_all),
        "obs_id": np.arange(n),
        "plot_id": np.arange(n_plot),
        "query_id": np.arange(n_query),
        "x_dim": np.arange(1),
    }

    with pm.Model(coords=coords) as model:

        # ---- Data ----

        # Age data (standardised) for all points (obs + plot + query)
        X_all_z_data = pm.Data("X_all_z", X_all_z, dims=("all_id", "x_dim"))

        # Ages (in months) for plotting/reporting (observed, predictive plots and specific query points)
        _ = pm.Data("X_obs", context.model_data.X_obs.flatten(), dims=("obs_id",))
        _ = pm.Data("X_plot", X_plot.flatten(), dims=("plot_id",))
        _ = pm.Data("X_query", X_query.flatten(), dims=("query_id",))

        # ---- Mean developmental trajectory ----

        # Anchors
        p_slope_low = context.model_config.p_slope_low_dist.to_pymc("p_slope_low")
        p_slope_hi = context.model_config.p_slope_hi_dist.to_pymc("p_slope_hi")

        # Slope (β₁)
        beta_1 = pm.Deterministic(
            "slope",
            (pymc_utils.logit(p_slope_hi) - pymc_utils.logit(p_slope_low))
            / (slope_age_b_z - slope_age_a_z),
        )

        # Intercept (β₀)
        beta_0 = pm.Deterministic(
            "intercept", pymc_utils.logit(p_slope_low) - beta_1 * slope_age_a_z
        )

        # Linear trend, evaluated at all input points (X_all_data)
        mean_trend_all = beta_0 + beta_1 * X_all_z_data[:, 0]  # (n_all,)

        # ---- Gaussian Process ----

        # GP hyperparameters

        # Length-scale ℓ (on the z-score scale)
        ell_unit = context.model_config.ell_unit_dist.to_pymc("ell_unit")
        ell = pm.Deterministic("ell", ell_low_z + (ell_high_z - ell_low_z) * ell_unit)

        # Amplitude (marginal standard deviation)
        eta = context.model_config.eta_dist.to_pymc("eta")

        # * HSGP specification *

        # co-variance function: radial basis function (RBF)
        cov0 = pm.gp.cov.ExpQuad(1, ls=ell)

        # unit-variance kernel: the amplitude η is applied to the basis function weights, not the kernel
        hsgp0 = pm.gp.HSGP(cov_func=cov0, m=M, L=L)

        # GP function values at all input points (obs + plot + query)
        g_unit = hsgp0.prior("g_unit", X=X_all_z_data, dims="all_id")

        # scale the raw GP values by the amplitude η to get the actual GP function values
        g = pm.Deterministic("g", eta * g_unit, dims=("all_id",))

        # the overall function is the sum of the mean trend and the GP deviation from that trend
        f_all = pm.Deterministic("f_all", mean_trend_all + g, dims=("all_id",))

        # Slice for obs / plot / query
        f_obs = pm.Deterministic("f_obs", f_all[i_obs0:i_obs1], dims=("obs_id",))
        f_plot = pm.Deterministic("f_plot", f_all[i_plot0:i_plot1], dims=("plot_id",))
        f_query = pm.Deterministic(
            "f_query", f_all[i_query0:i_query1], dims=("query_id",)
        )

        p_obs = pm.Deterministic("p_obs", pm.math.sigmoid(f_obs), dims=("obs_id",))
        _ = pm.Deterministic("p_plot", pm.math.sigmoid(f_plot), dims=("plot_id",))
        _ = pm.Deterministic("p_query", pm.math.sigmoid(f_query), dims=("query_id",))

        # z values for each slice (these are just the standardized ages)
        z_obs = pm.Deterministic(
            "z_obs", X_all_z_data[i_obs0:i_obs1, 0], dims=("obs_id",)
        )
        z_plot = pm.Deterministic(
            "z_plot", X_all_z_data[i_plot0:i_plot1, 0], dims=("plot_id",)
        )
        z_query = pm.Deterministic(
            "z_query", X_all_z_data[i_query0:i_query1, 0], dims=("query_id",)
        )

        # ---- Dispersion / overdispersion ----

        # dispersion parameter κ (kappa) is a function of age (z), with a minimum value and an
        # exponential increase/decrease with age

        # minimum kappa value
        kappa_min = context.model_config.kappa_min_dist.to_pymc("kappa_min")

        # slope of kappa with age (on log scale)
        a_kappa = context.model_config.a_kappa_dist.to_pymc("a_kappa")
        # magnitude of slope (on log scale)
        b_kappa_mag = context.model_config.b_kappa_mag_dist.to_pymc("b_kappa_mag")
        # the slope is negative, so we take the negative of the magnitude
        b_kappa = pm.Deterministic("b_kappa", -b_kappa_mag)

        # function to compute kappa as a function of age (z)
        def kappa_of_z(z):
            return kappa_min + pm.math.exp(a_kappa + b_kappa * z)

        # compute kappa for the observed data points, and use that in the likelihood
        kappa_obs = pm.Deterministic("kappa_obs", kappa_of_z(z_obs), dims="obs_id")

        # compute kappa for the plot and query points (for reporting, not used in likelihood)
        _ = pm.Deterministic("kappa_plot", kappa_of_z(z_plot), dims="plot_id")
        _ = pm.Deterministic("kappa_query", kappa_of_z(z_query), dims="query_id")

        # clip p_obs numerical issues when alpha or beta become extremely close to 0
        p_obs_clip = pm.math.clip(
            p_obs, math_constants.EPSILON, 1 - math_constants.EPSILON
        )

        # compute alpha and beta parameters for the Beta-Binomial likelihood based on p_obs and kappa_obs
        alpha_obs = p_obs_clip * kappa_obs
        beta_obs = (1 - p_obs_clip) * kappa_obs

        # Beta-binomial likelihood
        _ = pm.BetaBinomial(
            "y_obs",
            n=context.model_data.n_trials,
            alpha=alpha_obs,
            beta=beta_obs,
            observed=context.model_data.y_obs,
            dims=("obs_id",),
        )

    variables = pymc_utils.get_variables_dict(model)

    pymc_utils.report_model_summary(model)

    digraph = pymc_utils.model_to_graphviz(model)

    digraph.render(
        filename=os.path.join(context.reporting.output_dir, "gp_model_graph"),
        format="svg",
        cleanup=True,
    )

    context.set_model(model, variables)


def get_hsgp_hyperparams(
    X_obs_z,
    ell_range_z,
):
    """
    Compute HSGP basis size (m), boundary factor (c), and derived L, m.

    Parameters
    ----------
    X_obs_z : np.ndarray
        Standardised observed ages, shape (n, 1).
    ell_range_z : tuple of float
        Length-scale range in z-score scale.

    Returns
    -------
    tuple[list[float], list[int]]
        ``(L, M)`` where ``L`` is the HSGP boundary factor and ``M`` is the
        basis size, each wrapped in a single-element list (one per input dim).
    """
    x_min = float(np.min(X_obs_z))
    x_max = float(np.max(X_obs_z))

    ell_low_z = ell_range_z[0]
    ell_high_z = ell_range_z[1]

    m, c = pm.gp.hsgp_approx.approx_hsgp_hyperparams(
        x_range=[x_min, x_max],
        lengthscale_range=[ell_low_z, ell_high_z],
        cov_func="expquad",
    )

    S = max(abs(x_min), abs(x_max))
    L = [S * c]
    M = [m]

    return L, M


def extract_model_samples(trace: InferenceData) -> ModelSamples:
    """
    Extract model samples into a structured format for plotting and reporting.
    """

    # Posterior samples of the latent linear predictor f: (n, n_samples)
    f_obs = np.array(
        trace.posterior["f_obs"]
        .stack(sample=("chain", "draw"))
        .transpose("obs_id", "sample")
        .values
    )

    # Posterior samples of the latent linear predictor f for the plot points: (n_plot, n_samples)
    f_plot = np.array(
        trace.posterior["f_plot"]
        .stack(sample=("chain", "draw"))
        .transpose("plot_id", "sample")
        .values
    )

    # Posterior samples of the latent linear predictor f for the query points: (n_query, n_samples)
    f_query = np.array(
        trace.posterior["f_query"]
        .stack(sample=("chain", "draw"))
        .transpose("query_id", "sample")
        .values
    )

    p_obs = np.array(
        trace.posterior["p_obs"]
        .stack(sample=("chain", "draw"))
        .transpose("obs_id", "sample")
        .values
    )

    # Posterior samples of p(z) for the plot points: (n_plot, n_samples)
    p_plot = np.array(
        trace.posterior["p_plot"]
        .stack(sample=("chain", "draw"))
        .transpose("plot_id", "sample")
        .values
    )

    # Posterior samples of p(z) for the query points: (n_query, n_samples)
    p_query = np.array(
        trace.posterior["p_query"]
        .stack(sample=("chain", "draw"))
        .transpose("query_id", "sample")
        .values
    )

    # Posterior samples of kappa for the plot points: (n_plot, n_samples)
    kappa_plot = np.array(
        trace.posterior["kappa_plot"]
        .stack(sample=("chain", "draw"))
        .transpose("plot_id", "sample")
        .values
    )

    # Posterior samples of kappa for the query points: (n_query, n_samples)
    kappa_query = np.array(
        trace.posterior["kappa_query"]
        .stack(sample=("chain", "draw"))
        .transpose("query_id", "sample")
        .values
    )

    y_obs = np.array(trace.observed_data["y_obs"].values, dtype=int)

    # Posterior predictive samples of Y for the plot points: (n_plot, n_samples)
    y_plot = np.array(
        trace.posterior_predictive["y_plot"]
        .stack(sample=("chain", "draw"))
        .transpose("plot_id", "sample")
        .values,
        dtype=int,
    )

    # Posterior predictive samples of Y for the query points: (n_query, n_samples)
    y_query = np.array(
        trace.posterior_predictive["y_query"]
        .stack(sample=("chain", "draw"))
        .transpose("query_id", "sample")
        .values,
        dtype=int,
    )

    X_obs = np.array(trace.constant_data["X_obs"].values)

    X_plot = np.array(trace.constant_data["X_plot"].values)

    X_query = np.array(trace.constant_data["X_query"].values)

    X_obs_z = np.array(
        trace.posterior["z_obs"]
        .stack(sample=("chain", "draw"))
        .transpose("obs_id", "sample")
        .values
    )

    X_plot_z = np.array(
        trace.posterior["z_plot"]
        .stack(sample=("chain", "draw"))
        .transpose("plot_id", "sample")
        .values
    )

    X_query_z = np.array(
        trace.posterior["z_query"]
        .stack(sample=("chain", "draw"))
        .transpose("query_id", "sample")
        .values
    )

    model_samples = ModelSamples(
        X_obs=X_obs,
        X_plot=X_plot,
        X_query=X_query,
        X_obs_z=X_obs_z,
        X_plot_z=X_plot_z,
        X_query_z=X_query_z,
        f_obs=f_obs,
        f_plot=f_plot,
        f_query=f_query,
        p_obs=p_obs,
        p_plot=p_plot,
        p_query=p_query,
        y_obs=y_obs,
        y_plot=y_plot,
        y_query=y_query,
        kappa_plot=kappa_plot,
        kappa_query=kappa_query,
    )

    return model_samples


def prepare_data(context: ModelFitContext):
    """
    Load and prepare data. Report descriptive statistics.
    """
    print(
        "\n[green]------------------------------------------------------------[/green]"
    )
    print("[bold green]Prepare data[/bold green]")
    print("[green]------------------------------------------------------------[/green]")
    print()

    data, analysis_df, desc_stats = prepare_model_data(
        x_col="age", y_col="spoken", max_age_months=None, n_trials=800
    )

    context.set_model_data(data, analysis_df)
    context.dataframes["descriptive_stats"] = desc_stats

    desc_stats.to_csv(
        os.path.join(context.reporting.output_dir, "descriptive_statistics.csv"),
        index=True,
    )


def configure_model(context: ModelFitContext):
    """
    Configure priors and hyperparameters.
    """
    print(
        "\n[green]------------------------------------------------------------[/green]"
    )
    print("[bold green]Priors and hyperparameters[/bold green]")
    print("[green]------------------------------------------------------------[/green]")
    print()

    # Length scale
    ell_unit_dist = pz.Beta(alpha=3.0, beta=3.0)
    context.plots["ell_unit_dist"] = plot_dist.plot_distribution(
        ell_unit_dist, context.reporting.output_dir, "ell_unit_dist"
    )
    print(
        f"[bold yellow]ell_unit_dist:[/bold yellow] {ell_unit_dist.summary(mass=context.reporting.hdi)}"
    )

    # Amplitude
    eta_dist = pz.HalfNormal(sigma=0.4)
    context.plots["eta_dist"] = plot_dist.plot_distribution(
        eta_dist, context.reporting.output_dir, "eta_dist"
    )
    print(
        f"[bold yellow]eta_dist:[/bold yellow]: {eta_dist.summary(mass=context.reporting.hdi)}"
    )

    # Slope
    slope_age_low = 24
    slope_age_hi = 84

    p_slope_low_dist = pz.Beta(alpha=1.0, beta=15)
    context.plots["p_slope_low_dist"] = plot_dist.plot_distribution(
        p_slope_low_dist, context.reporting.output_dir, "p_slope_low_dist"
    )
    print(
        f"[bold yellow]p_slope_low_dist:[/bold yellow] {p_slope_low_dist.summary(mass=context.reporting.hdi)}"
    )

    p_slope_hi_dist = pz.Beta(alpha=1.1, beta=1.1)
    context.plots["p_slope_hi_dist"] = plot_dist.plot_distribution(
        p_slope_hi_dist, context.reporting.output_dir, "p_slope_hi_dist"
    )
    print(
        f"[bold yellow]p_slope_hi_dist:[/bold yellow] {p_slope_hi_dist.summary(mass=context.reporting.hdi)}"
    )

    # Dispersion / overdispersion

    kappa_min_dist = pz.LogNormal(mu=np.log(5.0), sigma=0.6)
    context.plots["kappa_min_dist"] = plot_dist.plot_distribution(
        kappa_min_dist, context.reporting.output_dir, "kappa_min_dist"
    )
    print(
        f"[bold yellow]kappa_min_dist:[/bold yellow] {kappa_min_dist.summary(mass=context.reporting.hdi)}"
    )

    a_kappa_dist = pz.Normal(mu=np.log(8.0), sigma=1.0)
    context.plots["a_kappa_dist"] = plot_dist.plot_distribution(
        a_kappa_dist, context.reporting.output_dir, "a_kappa_dist"
    )
    print(
        f"[bold yellow]a_kappa_dist:[/bold yellow] {a_kappa_dist.summary(mass=context.reporting.hdi)}"
    )

    b_kappa_mag_dist = pz.HalfNormal(sigma=0.3)
    context.plots["b_kappa_mag_dist"] = plot_dist.plot_distribution(
        b_kappa_mag_dist, context.reporting.output_dir, "b_kappa_mag_dist"
    )
    print(
        f"[bold yellow]b_kappa_mag_dist:[/bold yellow] {b_kappa_mag_dist.summary(mass=context.reporting.hdi)}"
    )

    # ------------------------------------------------------------
    # Model definition and initialisation
    # ------------------------------------------------------------

    config = ModelConfiguration(
        slope_anchors=(slope_age_low, slope_age_hi),
        ell_months_range=(2, 12),
        p_slope_low_dist=p_slope_low_dist,
        p_slope_hi_dist=p_slope_hi_dist,
        ell_unit_dist=ell_unit_dist,
        eta_dist=eta_dist,
        kappa_min_dist=kappa_min_dist,
        a_kappa_dist=a_kappa_dist,
        b_kappa_mag_dist=b_kappa_mag_dist,
        n_plot=500,
        ages_query=[12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90],
    )

    context.set_model_config(config)


def prior_predictive_checks(context: ModelFitContext):
    """
    Run prior predictive checks: sample from the prior, and visualise the implied prior predictive distribution.
    """
    print(
        "\n[green]------------------------------------------------------------[/green]"
    )
    print("[bold green]Prior predictive checks[/bold green]")
    print("[green]------------------------------------------------------------[/green]")
    print()

    with context.model:
        prior_samples = pm.sample_prior_predictive(
            draws=2000, random_seed=context.sampling.random_seed
        )

    # Sample prior functions

    p_plot_samples = (
        prior_samples.prior["p_plot"]
        .stack(sample=("chain", "draw"))
        .transpose("plot_id", "sample")  # (n_plot, n_samples)
    )

    context.set_prior_samples(prior_samples)

    plotting.plot_prior_samples(
        prior_samples.constant_data["X_plot"].values,
        p_plot_samples.values,
        context.analysis_df["age"],
        context.analysis_df["spoken"],
        n_trials=context.model_data.n_trials,
        n_curves=1000,
        x_label="Age (months)",
        y_label="Words spoken",
        filename="prior_samples",
        output_dir=context.reporting.output_dir,
    )

    y_pred = prior_samples.prior_predictive["y_obs"].stack(sample=("chain", "draw"))
    obs_ages = prior_samples.constant_data["X_obs"].values

    plotting.plot_prior_predictions(
        obs_ages,
        y_pred,
        context.analysis_df["age"],
        context.analysis_df["spoken"],
        n_trials=context.model_data.n_trials,
        x_label="Age (months)",
        y_label="Words spoken (predicted)",
        filename="prior_predictions",
        output_dir=context.reporting.output_dir,
    )

    # Prior predictive distribution

    plot_predictive.plot_prior_predictive_checks(
        prior_samples,
        random_seed=context.sampling.random_seed,
        output_dir=context.reporting.output_dir,
        filename="prior_predictive_checks",
    )


def sample(context: ModelFitContext):
    """
    Draw samples from the posterior using MCMC.
    """
    print(
        "\n[green]------------------------------------------------------------[/green]"
    )
    print("[bold green]Posterior sampling[/bold green]")
    print("[green]------------------------------------------------------------[/green]")
    print()

    print(context.sampling)
    print()

    with context.model:
        trace = pm.sample(
            context.sampling.draws,
            tune=context.sampling.tune,
            chains=context.sampling.chains,
            cores=context.sampling.cores,
            target_accept=context.sampling.target_accept,
            nuts_sampler="nutpie",
            return_inferencedata=True,
            random_seed=context.sampling.random_seed,
        )

    context.set_trace(trace)

    print()
    print()
    print("[bold green]Posterior sampling completed.[/bold green]")


def diagnostics(context: ModelFitContext):
    """
    Run diagnostics on the posterior samples, including convergence diagnostics and posterior predictive checks.
    """
    print(
        "\n[green]------------------------------------------------------------[/green]"
    )
    print("[bold green]Diagnostics[/bold green]")
    print("[green]------------------------------------------------------------[/green]")
    print()

    # Summary diagnostic statistics

    var_names = [
        var.name for var in context.model.unobserved_RVs if var.size.eval() <= 2
    ]
    diagnostics_df = az.summary(
        context.trace, var_names=var_names, round_to=3, hdi_prob=context.reporting.hdi
    )

    diagnostics_df.to_csv(
        os.path.join(context.reporting.output_dir, "diagnostics.csv"), index=True
    )

    pprint(diagnostics_df)

    # Kernel density estimates (KDE) of the joint posterior, and marginals

    plot_diagnostics_mcmc.plot_kde_pair(
        context.trace,
        var_names=var_names,
        output_dir=context.reporting.output_dir,
        filename="pair_plot",
    )
    context.plots["pair_plot"] = plt.gcf()
    plt.close()

    # Trace plot

    var_names_ext = var_names + ["kappa_obs"]

    az.plot_trace(
        context.trace,
        combined=True,
        var_names=var_names_ext,
    )
    plt.savefig(os.path.join(context.reporting.output_dir, "trace_plot.png"), dpi=300)
    context.plots["trace_plot"] = plt.gcf()
    plt.close()

    # Energy transition distribution
    az.plot_energy(context.trace, figsize=plot_styles.FIGSIZE_SM)
    plt.savefig(os.path.join(context.reporting.output_dir, "energy_plot.png"), dpi=300)
    context.plots["energy_plot"] = plt.gcf()
    plt.close()

    # Posterior densities
    az.plot_posterior(
        context.trace.posterior,
        var_names=var_names,
        point_estimate="median",
        hdi_prob=context.reporting.hdi,
    )
    plt.savefig(
        os.path.join(context.reporting.output_dir, "posterior_plot.png"), dpi=300
    )
    context.plots["posterior_plot"] = plt.gcf()
    plt.close()

    # Pareto-smoothed importance sampling

    with context.model:
        trace = pm.compute_log_likelihood(context.trace)

    context.set_trace(trace)

    loocv = az.loo(context.trace)
    context.set_loocv(loocv)
    print(loocv)


def sample_posterior_predictive(context: ModelFitContext):
    """
    Sample from the posterior predictive distribution.
    """
    print(
        "\n[green]------------------------------------------------------------[/green]"
    )
    print("[bold green]Posterior predictions[/bold green]")
    print("[green]------------------------------------------------------------[/green]")
    print()

    p_plot = context.model_variables["p_plot"]
    p_query = context.model_variables["p_query"]
    kappa_plot = context.model_variables["kappa_plot"]
    kappa_query = context.model_variables["kappa_query"]

    with context.model:
        p_plot_clip = pm.math.clip(
            p_plot, math_constants.EPSILON, 1 - math_constants.EPSILON
        )
        alpha_plot = p_plot_clip * kappa_plot
        beta_plot = (1 - p_plot_clip) * kappa_plot
        pm.BetaBinomial(
            "y_plot",
            n=context.model_data.n_trials,
            alpha=alpha_plot,
            beta=beta_plot,
            dims=("plot_id",),
        )
        p_query_clip = pm.math.clip(
            p_query, math_constants.EPSILON, 1 - math_constants.EPSILON
        )
        alpha_query = p_query_clip * kappa_query
        beta_query = (1 - p_query_clip) * kappa_query
        pm.BetaBinomial(
            "y_query",
            n=context.model_data.n_trials,
            alpha=alpha_query,
            beta=beta_query,
            dims=("query_id",),
        )
        trace = pm.sample_posterior_predictive(
            context.trace,
            var_names=["y_plot", "y_query", "y_obs"],
            extend_inferencedata=True,
            random_seed=context.sampling.random_seed,
        )

    context.set_trace(trace)

    trace.to_netcdf(os.path.join(context.reporting.output_dir, "trace.nc"))

    sample_data = extract_model_samples(context.trace)
    context.set_model_samples(sample_data)


def posterior_summary(context: ModelFitContext):
    """
    Compute and store the posterior summary table at query ages.
    """

    posterior_summary_df = posterior_analysis.posterior_summary_table(
        context.model_samples.X_query,
        context.model_samples.p_query,
        context.model_samples.y_query,
        n_trials=context.model_data.n_trials,
        hdi_prob=context.reporting.hdi,
    )

    pprint(posterior_summary_df)

    context.dataframes["posterior_summary"] = posterior_summary_df

    posterior_summary_df.to_csv(
        os.path.join(context.reporting.output_dir, "posterior_summary.csv"), index=False
    )


def report(context: ModelFitContext):
    """
    Copy output artefacts to the report directory.
    """

    REPORT_OUTPUT_DIR = os.path.join(
        local_env.REPORT_FIGS_DIR, context.reporting.model_label
    )

    if os.path.exists(REPORT_OUTPUT_DIR):
        shutil.rmtree(REPORT_OUTPUT_DIR)

    os.makedirs(REPORT_OUTPUT_DIR, exist_ok=True)

    print()
    print(f"Output: {context.reporting.output_dir}")
    print(f"Report output: {REPORT_OUTPUT_DIR}")

    for filename in os.listdir(context.reporting.output_dir):
        if not (
            filename.endswith(".png")
            or filename.endswith(".svg")
            or filename.endswith(".csv")
        ):
            continue
        source_file = os.path.join(context.reporting.output_dir, filename)
        dest_file = os.path.join(REPORT_OUTPUT_DIR, filename)
        shutil.copy(source_file, dest_file)

    model_output_md_source = os.path.join(
        local_env.DOCS_DIR, "models", context.reporting.model_name.lower(), "index.qmd"
    )

    model_output_md_dest = os.path.join(context.reporting.output_dir, "index.qmd")

    if os.path.exists(model_output_md_source):
        shutil.copy(model_output_md_source, model_output_md_dest)
    else:
        raise FileNotFoundError(
            f"Source model output markdown file not found: {model_output_md_source}"
        )

    print(f"[bold green]Report written to: {model_output_md_dest}[/bold green]")

    print(
        f"\n[bold yellow]To render, execute:[/bold yellow] [blue]quarto render {model_output_md_dest}[/blue]"
    )

    print(
        f"\n[bold yellow]For live preview (editing), execute:[/bold yellow] [blue]quarto preview {model_output_md_dest}[/blue]"
    )


def fit(config: str) -> ModelFitContext:
    print(
        "\n[green]============================================================[/green]"
    )
    print(
        "[bold green]Fitting Model VG01: Influence of age on words spoken (A → S)[/bold green]"
    )
    print("[green]============================================================[/green]")
    print()

    env_info.report_environment_info()

    package_list = [
        "arviz",
        "matplotlib",
        "numba",
        "numpy",
        "numpyro",
        "pandas",
        "pymc",
        "pytensor",
    ]

    print()
    package_metadata.report_package_versions(package_list)

    context = ModelFitContext(
        reporting=reporting.ReportingConfiguration(
            model_name="VG01",
            config_name="age-spoken-ds",
            output_root_dir=local_env.OUTPUT_DIR,
            hdi=0.90,
        ),
        sampling=sampling.get_sampling_configuration(config),
    )

    if os.path.exists(context.reporting.output_dir):
        shutil.rmtree(context.reporting.output_dir)

    os.makedirs(context.reporting.output_dir, exist_ok=True)

    prepare_data(context)

    configure_model(context)

    build_model(context)

    prior_predictive_checks(context)

    sample(context)

    diagnostics(context)

    sample_posterior_predictive(context)

    posterior_summary(context)

    plotting.plot_posterior_predictive_count_distributions_by_query_age(
        X_query=context.model_samples.X_query,
        y_query=context.model_samples.y_query,
        n_trials=context.model_data.n_trials,
        output_dir=context.reporting.output_dir,
        filename="posterior_predictive_count_distributions",
    )

    plotting.plot_posterior_predictive_pmf(
        context.model_samples.X_query,
        context.model_samples.X_plot,
        context.model_samples.y_plot,
        context.model_data.n_trials,
        output_dir=context.reporting.output_dir,
        filename="posterior_predictive_pmf",
    )

    plotting.plot_posterior_predictive_cdf(
        context.model_samples.X_query,
        context.model_samples.X_plot,
        context.model_samples.y_plot,
        context.model_data.n_trials,
        output_dir=context.reporting.output_dir,
        filename="posterior_predictive_cdf",
    )

    plotting.plot_posterior_predictive_median_trend(
        context.model_samples.X_plot,
        context.model_samples.y_plot,
        context.model_samples.X_obs,
        context.model_samples.y_obs,
        output_dir=context.reporting.output_dir,
        filename="plot_posterior_predictive_median_trend",
    )

    plotting.plot_posterior_predictive_median_trend(
        context.model_samples.X_plot,
        context.model_samples.y_plot,
        context.model_samples.X_obs,
        context.model_samples.y_obs,
        smooth=True,
        savgol_window_length=15,
        savgol_polyorder=3,
        smooth_intervals=True,
        output_dir=context.reporting.output_dir,
        filename="plot_posterior_predictive_median_trend_smoothed",
    )

    plotting.plot_expected_learning_rate(
        context.model_samples.X_plot,
        context.model_samples.f_plot,
        n_trials=context.model_data.n_trials,
        hdi_prob=context.reporting.hdi,
        output_dir=context.reporting.output_dir,
        filename="expected_learning_rate",
    )

    plotting.plot_expected_learning_rate(
        context.model_samples.X_plot,
        context.model_samples.f_plot,
        n_trials=context.model_data.n_trials,
        hdi_prob=context.reporting.hdi,
        smooth=True,
        savgol_window_length=15,
        savgol_polyorder=3,
        smooth_intervals=True,
        output_dir=context.reporting.output_dir,
        filename="expected_learning_rate_smoothed",
    )

    plotting.plot_posterior_kappa(
        context.model_samples.X_plot,
        context.model_samples.kappa_plot,
        context.model_samples.X_query,
        context.model_samples.kappa_query,
        n_trials=context.model_data.n_trials,
        hdi_prob=context.reporting.hdi,
        output_dir=context.reporting.output_dir,
        filename="posterior_kappa",
    )

    report(context)

    return context
