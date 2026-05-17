# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Shared dataclasses and pipeline functions for the bivariate vocabulary growth
models (VG05, VG06).

Uses a production-ratio reparameterization:
    p_U(a) = sigmoid(f_U(a))
    q(a)   = sigmoid(h(a))       # fraction of understood words spoken
    p_S(a) = p_U(a) * q(a)       # enforces p_S <= p_U by construction
"""

import os
import shutil
import time
from dataclasses import dataclass

import arviz as az
import dse_research_utils.environment.info as env_info
import dse_research_utils.math.constants as math_constants
import dse_research_utils.metadata.packages as package_metadata
import dse_research_utils.plot.diagnostics_mcmc as plot_diagnostics_mcmc
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
import xarray as xr
from preliz.distributions.distributions import Continuous

import vocab_growth.data_utils as vocab_data_utils
import vocab_growth.environment as local_env
import vocab_growth.plotting as plotting
import vocab_growth.posterior_analysis as posterior_analysis
import vocab_growth.reporting as vg_reporting
from vocab_growth.models.common import (
    PACKAGE_LIST,
    BaseModelConfiguration,
    ModelFitContext,
    _plot_and_print_dist,
    _report_diagnostic_warnings,
    get_hsgp_hyperparams,
    report,
)
from vocab_growth.models.definitions import BivariateModelDefinition
from vocab_growth.models.diagnostics_utils import pair_plot_rc_params
from vocab_growth.plotting import _save_csv
from vocab_growth.reporting import (
    config_table,
    console,
    dataframe_table,
    heading,
    key_value_table,
    pipeline_summary,
    run_banner,
    section,
)

EPSILON = math_constants.EPSILON


# ============================================================
# Bivariate-specific dataclasses
# ============================================================


@dataclass
class BivariateModelConfiguration(BaseModelConfiguration):
    """Configuration for the bivariate (understood + spoken) model."""

    # Understood (U) trajectory priors
    p_slope_low_u_dist: Continuous
    p_slope_hi_u_dist: Continuous
    ell_unit_u_dist: Continuous
    eta_u_dist: Continuous

    # Production ratio (q) priors
    p_slope_low_q_dist: Continuous
    p_slope_hi_q_dist: Continuous
    ell_unit_q_dist: Continuous
    eta_q_dist: Continuous

    # Kappa priors — understood
    kappa_min_u_dist: Continuous
    a_kappa_u_dist: Continuous
    b_kappa_mag_u_dist: Continuous

    # Kappa priors — spoken
    kappa_min_s_dist: Continuous
    a_kappa_s_dist: Continuous
    b_kappa_mag_s_dist: Continuous


@dataclass
class BivariateModelSamples:
    """Posterior and predictive samples from the bivariate model."""

    # Shared age grids
    X_obs: np.ndarray
    """Observed ages in months, shape (n,)."""
    X_plot: np.ndarray
    """Ages in months for the plot points, shape (n_plot,)."""
    X_query: np.ndarray
    """Ages in months for the query points, shape (n_query,)."""

    X_obs_z: np.ndarray
    """Standardized observed ages, shape (n, n_samples)."""
    X_plot_z: np.ndarray
    """Standardized ages for the plot points, shape (n_plot, n_samples)."""
    X_query_z: np.ndarray
    """Standardized ages for the query points, shape (n_query, n_samples)."""

    # Understood (U) samples
    f_u_obs: np.ndarray
    f_u_plot: np.ndarray
    f_u_query: np.ndarray
    p_u_obs: np.ndarray
    p_u_plot: np.ndarray
    p_u_query: np.ndarray
    y_u_obs: np.ndarray
    y_u_plot: np.ndarray
    y_u_query: np.ndarray
    kappa_u_plot: np.ndarray
    kappa_u_query: np.ndarray

    # Production rate (q) samples
    h_obs: np.ndarray
    h_plot: np.ndarray
    h_query: np.ndarray
    q_obs: np.ndarray
    q_plot: np.ndarray
    q_query: np.ndarray

    # Spoken (S) samples (derived)
    f_s_obs: np.ndarray
    f_s_plot: np.ndarray
    f_s_query: np.ndarray
    p_s_obs: np.ndarray
    p_s_plot: np.ndarray
    p_s_query: np.ndarray
    y_s_obs: np.ndarray
    y_s_plot: np.ndarray
    y_s_query: np.ndarray
    kappa_s_plot: np.ndarray
    kappa_s_query: np.ndarray

    # Observation masks (over obs_id)
    obs_u_mask: np.ndarray
    """Boolean array: True where understood is observed, shape (n,)."""
    obs_s_mask: np.ndarray
    """Boolean array: True where spoken is observed, shape (n,)."""


BivariateContext = ModelFitContext[BivariateModelConfiguration, BivariateModelSamples]


# ============================================================
# Data preparation
# ============================================================


def prepare_bivariate_data(
    context: BivariateContext,
    definition: BivariateModelDefinition,
):
    """Load and prepare data for a bivariate model from its definition."""
    df = vocab_data_utils.load_data(
        population=definition.population,
        columns=["age", "understood", "spoken"],
        sample_fraction=definition.sample_fraction,
        random_seed=definition.random_seed,
    )
    analysis_df = df[["age", "understood", "spoken"]].copy()

    # Keep rows where at least one outcome is observed (and age is present)
    analysis_df = analysis_df.dropna(subset=["age"])
    has_u = analysis_df["understood"].notna()
    has_s = analysis_df["spoken"].notna()
    analysis_df = analysis_df[has_u | has_s].reset_index(drop=True)

    desc = descriptive_stats.describe_all(analysis_df, alpha=0.05)

    n = len(analysis_df)
    n_u = int(analysis_df["understood"].notna().sum())
    n_s = int(analysis_df["spoken"].notna().sum())
    n_both = int(
        (analysis_df["understood"].notna() & analysis_df["spoken"].notna()).sum()
    )

    key_value_table(
        "Observation counts",
        [
            ("Total observations", n),
            ("Understood observed", n_u),
            ("Spoken observed", n_s),
            ("Both observed", n_both),
            ("Understood only", n_u - n_both),
            ("Spoken only", n_s - n_both),
        ],
    )
    dataframe_table(desc, title="Descriptive statistics")

    # Create a BinomialModelData for the context interface (using understood as primary)
    X_obs = np.asarray(analysis_df["age"], dtype=float).reshape(-1, 1)
    y_u_valid = analysis_df.loc[analysis_df["understood"].notna(), "understood"]
    y_obs_placeholder = np.zeros(n, dtype=int)
    y_obs_placeholder[analysis_df["understood"].notna().values] = (
        y_u_valid.values.astype(int)
    )

    bmd = model_data.BinomialModelData(
        X_obs=X_obs, y_obs=y_obs_placeholder, n_trials=definition.n_trials
    )

    context.set_model_data(bmd, analysis_df)
    context.dataframes["descriptive_stats"] = desc

    desc.to_csv(
        os.path.join(context.reporting.output_dir, "descriptive_statistics.csv"),
        index=True,
    )


# ============================================================
# Prior configuration
# ============================================================


def configure_bivariate_priors(
    context: BivariateContext,
    definition: BivariateModelDefinition,
):
    """Configure priors and hyperparameters from a bivariate model definition."""
    heading("Understood trajectory priors", style="bold cyan")
    # --- Understood (U) trajectory priors ---

    ell_unit_u_dist = pz.Beta(
        alpha=definition.ell_unit_u_alpha, beta=definition.ell_unit_u_beta
    )
    _plot_and_print_dist(context, ell_unit_u_dist, "ell_unit_u_dist")

    eta_u_dist = pz.HalfNormal(sigma=definition.eta_u_sigma)
    _plot_and_print_dist(context, eta_u_dist, "eta_u_dist")

    # Slope priors for understood
    p_slope_low_u_dist = pz.Beta(
        alpha=definition.p_slope_low_u_alpha, beta=definition.p_slope_low_u_beta
    )
    _plot_and_print_dist(context, p_slope_low_u_dist, "p_slope_low_u_dist")

    p_slope_hi_u_dist = pz.Beta(
        alpha=definition.p_slope_hi_u_alpha, beta=definition.p_slope_hi_u_beta
    )
    _plot_and_print_dist(context, p_slope_hi_u_dist, "p_slope_hi_u_dist")

    # --- Production ratio (q) priors ---
    heading("Production ratio priors", style="bold cyan")

    ell_unit_q_dist = pz.Beta(
        alpha=definition.ell_unit_q_alpha, beta=definition.ell_unit_q_beta
    )
    _plot_and_print_dist(context, ell_unit_q_dist, "ell_unit_q_dist")

    eta_q_dist = pz.HalfNormal(sigma=definition.eta_q_sigma)
    _plot_and_print_dist(context, eta_q_dist, "eta_q_dist")

    p_slope_low_q_dist = pz.Beta(
        alpha=definition.p_slope_low_q_alpha, beta=definition.p_slope_low_q_beta
    )
    _plot_and_print_dist(context, p_slope_low_q_dist, "p_slope_low_q_dist")

    p_slope_hi_q_dist = pz.Beta(
        alpha=definition.p_slope_hi_q_alpha, beta=definition.p_slope_hi_q_beta
    )
    _plot_and_print_dist(context, p_slope_hi_q_dist, "p_slope_hi_q_dist")

    # --- Kappa priors — understood ---
    heading("Kappa priors — understood", style="bold cyan")

    kp_u = definition.kappa_u
    kappa_min_u_dist = pz.LogNormal(mu=kp_u.kappa_min_mu, sigma=kp_u.kappa_min_sigma)
    _plot_and_print_dist(context, kappa_min_u_dist, "kappa_min_u_dist")

    a_kappa_u_dist = pz.Normal(mu=kp_u.a_kappa_mu, sigma=kp_u.a_kappa_sigma)
    _plot_and_print_dist(context, a_kappa_u_dist, "a_kappa_u_dist")

    b_kappa_mag_u_dist = pz.HalfNormal(sigma=kp_u.b_kappa_mag_sigma)
    _plot_and_print_dist(context, b_kappa_mag_u_dist, "b_kappa_mag_u_dist")

    # --- Kappa priors — spoken ---
    heading("Kappa priors — spoken", style="bold cyan")

    kp_s = definition.kappa_s
    kappa_min_s_dist = pz.LogNormal(mu=kp_s.kappa_min_mu, sigma=kp_s.kappa_min_sigma)
    _plot_and_print_dist(context, kappa_min_s_dist, "kappa_min_s_dist")

    a_kappa_s_dist = pz.Normal(mu=kp_s.a_kappa_mu, sigma=kp_s.a_kappa_sigma)
    _plot_and_print_dist(context, a_kappa_s_dist, "a_kappa_s_dist")

    b_kappa_mag_s_dist = pz.HalfNormal(sigma=kp_s.b_kappa_mag_sigma)
    _plot_and_print_dist(context, b_kappa_mag_s_dist, "b_kappa_mag_s_dist")

    # --- Configuration object ---

    config = BivariateModelConfiguration(
        slope_anchors=definition.slope_anchors,
        ell_months_range=definition.ell_months_range,
        # Understood
        p_slope_low_u_dist=p_slope_low_u_dist,
        p_slope_hi_u_dist=p_slope_hi_u_dist,
        ell_unit_u_dist=ell_unit_u_dist,
        eta_u_dist=eta_u_dist,
        # Production rate
        p_slope_low_q_dist=p_slope_low_q_dist,
        p_slope_hi_q_dist=p_slope_hi_q_dist,
        ell_unit_q_dist=ell_unit_q_dist,
        eta_q_dist=eta_q_dist,
        # Kappa — understood
        kappa_min_u_dist=kappa_min_u_dist,
        a_kappa_u_dist=a_kappa_u_dist,
        b_kappa_mag_u_dist=b_kappa_mag_u_dist,
        # Kappa — spoken
        kappa_min_s_dist=kappa_min_s_dist,
        a_kappa_s_dist=a_kappa_s_dist,
        b_kappa_mag_s_dist=b_kappa_mag_s_dist,
        n_plot=definition.n_plot,
        ages_query=definition.ages_query,
    )

    context.set_model_config(config)


# ============================================================
# Model building
# ============================================================


def build_model(context: BivariateContext):
    """Build the bivariate PyMC model."""
    config = context.model_config

    analysis_df = context.analysis_df

    # Observation masks
    has_u = analysis_df["understood"].notna().values
    has_s = analysis_df["spoken"].notna().values

    X_obs = np.asarray(analysis_df["age"], dtype=float).reshape(-1, 1)
    y_u_observed = np.asarray(analysis_df.loc[has_u, "understood"], dtype=int)
    y_s_observed = np.asarray(analysis_df.loc[has_s, "spoken"], dtype=int)

    idx_u = np.where(has_u)[0]
    idx_s = np.where(has_s)[0]

    n = len(X_obs)
    n_u = len(y_u_observed)
    n_s = len(y_s_observed)
    n_trials = context.model_data.n_trials

    # Validate
    if not np.all(y_u_observed >= 0):
        raise ValueError("y_u contains negative counts.")
    if not np.all(y_u_observed <= n_trials):
        raise ValueError("y_u exceeds n_trials.")
    if not np.all(y_s_observed >= 0):
        raise ValueError("y_s contains negative counts.")
    if not np.all(y_s_observed <= n_trials):
        raise ValueError("y_s exceeds n_trials.")

    # Standardise ages
    X_obs_mean = float(np.mean(X_obs))
    X_obs_std = float(np.std(X_obs, ddof=1))

    if not np.isfinite(X_obs_std) or X_obs_std <= 0:
        raise ValueError("Age standard deviation must be positive.")

    key_value_table(
        "Build configuration",
        [
            ("Total observations", n),
            ("Understood observed", n_u),
            ("Spoken observed", n_s),
            ("n_trials", n_trials),
            ("Age mean (months)", X_obs_mean),
            ("Age std (months)", X_obs_std),
            ("Slope anchors (months)", config.slope_anchors),
            ("Length-scale range (months)", config.ell_months_range),
            ("Query ages (months)", config.ages_query),
        ],
    )

    X_obs_z = (X_obs - X_obs_mean) / X_obs_std

    # Plot grid
    X_plot = np.linspace(X_obs.min(), X_obs.max(), config.n_plot).reshape(-1, 1)
    X_plot_z = (X_plot - X_obs_mean) / X_obs_std

    # Query grid
    X_query = np.array(config.ages_query).reshape(-1, 1)
    X_query_z = (X_query - X_obs_mean) / X_obs_std

    # Stack all
    X_all_z = np.vstack([X_obs_z, X_plot_z, X_query_z])

    n_plot = X_plot_z.shape[0]
    n_query = X_query_z.shape[0]
    n_all = X_all_z.shape[0]

    # Length-scale bounds
    ell_low_months = float(config.ell_months_range[0])
    ell_high_months = float(config.ell_months_range[1])

    if ell_low_months <= 0 or ell_high_months <= 0:
        raise ValueError("Length-scale bounds must be positive (in months).")
    if ell_high_months <= ell_low_months:
        raise ValueError("ell_months_range must be (low, high) with high > low.")

    ell_low_z = ell_low_months / X_obs_std
    ell_high_z = ell_high_months / X_obs_std
    ell_range_z = (ell_low_z, ell_high_z)

    L, M = get_hsgp_hyperparams(X_obs_z, ell_range_z)

    # Slope anchors
    slope_age_a = float(config.slope_anchors[0])
    slope_age_b = float(config.slope_anchors[1])
    slope_age_a_z = (slope_age_a - X_obs_mean) / X_obs_std
    slope_age_b_z = (slope_age_b - X_obs_mean) / X_obs_std

    key_value_table(
        "Derived quantities",
        [
            ("HSGP basis size (m)", M),
            ("HSGP boundary factor (L)", L),
            ("Slope anchors (z-score)", (slope_age_a_z, slope_age_b_z)),
            ("Length-scale range (z-score)", (ell_low_z, ell_high_z)),
        ],
    )

    # Slice indices
    i_obs0, i_obs1 = 0, n
    i_plot0, i_plot1 = i_obs1, i_obs1 + n_plot
    i_query0, i_query1 = i_plot1, i_plot1 + n_query

    coords = {
        "all_id": np.arange(n_all),
        "obs_id": np.arange(n),
        "obs_u_id": np.arange(n_u),
        "obs_s_id": np.arange(n_s),
        "plot_id": np.arange(n_plot),
        "query_id": np.arange(n_query),
        "x_dim": np.arange(1),
    }

    with pm.Model(coords=coords) as model_pm:

        # ---- Data ----

        X_all_z_data = pm.Data("X_all_z", X_all_z, dims=("all_id", "x_dim"))

        _ = pm.Data("X_obs", X_obs.flatten(), dims=("obs_id",))
        _ = pm.Data("X_plot", X_plot.flatten(), dims=("plot_id",))
        _ = pm.Data("X_query", X_query.flatten(), dims=("query_id",))

        # Store masks and indices as constant data for extraction
        _ = pm.Data("obs_u_mask", has_u.astype(int), dims=("obs_id",))
        _ = pm.Data("obs_s_mask", has_s.astype(int), dims=("obs_id",))

        # ============================================================
        # Understood (U) trajectory: f_U(a) -> p_U(a)
        # ============================================================

        p_slope_low_u = config.p_slope_low_u_dist.to_pymc("p_slope_low_u")
        p_slope_hi_u = config.p_slope_hi_u_dist.to_pymc("p_slope_hi_u")

        slope_u = pm.Deterministic(
            "slope_u",
            (pymc_utils.logit(p_slope_hi_u) - pymc_utils.logit(p_slope_low_u))
            / (slope_age_b_z - slope_age_a_z),
        )
        intercept_u = pm.Deterministic(
            "intercept_u",
            pymc_utils.logit(p_slope_low_u) - slope_u * slope_age_a_z,
        )
        mean_trend_u = intercept_u + slope_u * X_all_z_data[:, 0]

        # GP for understood
        ell_unit_u = config.ell_unit_u_dist.to_pymc("ell_unit_u")
        ell_u = pm.Deterministic(
            "ell_u", ell_low_z + (ell_high_z - ell_low_z) * ell_unit_u
        )
        eta_u = config.eta_u_dist.to_pymc("eta_u")

        cov_u = pm.gp.cov.ExpQuad(1, ls=ell_u)
        hsgp_u = pm.gp.HSGP(cov_func=cov_u, m=M, L=L)
        g_unit_u = hsgp_u.prior("g_unit_u", X=X_all_z_data, dims="all_id")
        g_u = pm.Deterministic("g_u", eta_u * g_unit_u, dims=("all_id",))

        f_u_all = pm.Deterministic("f_u_all", mean_trend_u + g_u, dims=("all_id",))

        # ============================================================
        # Production ratio: h(a) -> q(a) = sigmoid(h(a))
        # ============================================================

        p_slope_low_q = config.p_slope_low_q_dist.to_pymc("p_slope_low_q")
        p_slope_hi_q = config.p_slope_hi_q_dist.to_pymc("p_slope_hi_q")

        slope_q = pm.Deterministic(
            "slope_q",
            (pymc_utils.logit(p_slope_hi_q) - pymc_utils.logit(p_slope_low_q))
            / (slope_age_b_z - slope_age_a_z),
        )
        intercept_q = pm.Deterministic(
            "intercept_q",
            pymc_utils.logit(p_slope_low_q) - slope_q * slope_age_a_z,
        )
        mean_trend_q = intercept_q + slope_q * X_all_z_data[:, 0]

        # GP for production rate
        ell_unit_q = config.ell_unit_q_dist.to_pymc("ell_unit_q")
        ell_q = pm.Deterministic(
            "ell_q", ell_low_z + (ell_high_z - ell_low_z) * ell_unit_q
        )
        eta_q = config.eta_q_dist.to_pymc("eta_q")

        cov_q = pm.gp.cov.ExpQuad(1, ls=ell_q)
        hsgp_q = pm.gp.HSGP(cov_func=cov_q, m=M, L=L)
        g_unit_q = hsgp_q.prior("g_unit_q", X=X_all_z_data, dims="all_id")
        g_q = pm.Deterministic("g_q", eta_q * g_unit_q, dims=("all_id",))

        h_all = pm.Deterministic("h_all", mean_trend_q + g_q, dims=("all_id",))

        # ============================================================
        # Derived quantities: p_U, q, p_S
        # ============================================================

        p_u_all = pm.Deterministic(
            "p_u_all", pm.math.sigmoid(f_u_all), dims=("all_id",)
        )
        q_all = pm.Deterministic("q_all", pm.math.sigmoid(h_all), dims=("all_id",))
        p_s_all = pm.Deterministic("p_s_all", p_u_all * q_all, dims=("all_id",))

        # f_S derived for diagnostics/plotting
        p_s_all_clip = pm.math.clip(p_s_all, EPSILON, 1 - EPSILON)
        f_s_all = pm.Deterministic(
            "f_s_all",
            pm.math.log(p_s_all_clip) - pm.math.log(1 - p_s_all_clip),
            dims=("all_id",),
        )

        # ---- Slice into obs/plot/query ----

        # Understood
        _ = pm.Deterministic("f_u_obs", f_u_all[i_obs0:i_obs1], dims=("obs_id",))
        _ = pm.Deterministic("f_u_plot", f_u_all[i_plot0:i_plot1], dims=("plot_id",))
        _ = pm.Deterministic(
            "f_u_query", f_u_all[i_query0:i_query1], dims=("query_id",)
        )

        p_u_obs = pm.Deterministic("p_u_obs", p_u_all[i_obs0:i_obs1], dims=("obs_id",))
        _ = pm.Deterministic("p_u_plot", p_u_all[i_plot0:i_plot1], dims=("plot_id",))
        _ = pm.Deterministic(
            "p_u_query", p_u_all[i_query0:i_query1], dims=("query_id",)
        )

        # Production rate
        _ = pm.Deterministic("h_obs", h_all[i_obs0:i_obs1], dims=("obs_id",))
        _ = pm.Deterministic("h_plot", h_all[i_plot0:i_plot1], dims=("plot_id",))
        _ = pm.Deterministic("h_query", h_all[i_query0:i_query1], dims=("query_id",))

        _ = pm.Deterministic("q_obs", q_all[i_obs0:i_obs1], dims=("obs_id",))
        _ = pm.Deterministic("q_plot", q_all[i_plot0:i_plot1], dims=("plot_id",))
        _ = pm.Deterministic("q_query", q_all[i_query0:i_query1], dims=("query_id",))

        # Spoken (derived)
        p_s_obs = pm.Deterministic("p_s_obs", p_s_all[i_obs0:i_obs1], dims=("obs_id",))
        _ = pm.Deterministic("p_s_plot", p_s_all[i_plot0:i_plot1], dims=("plot_id",))
        _ = pm.Deterministic(
            "p_s_query", p_s_all[i_query0:i_query1], dims=("query_id",)
        )

        _ = pm.Deterministic("f_s_obs", f_s_all[i_obs0:i_obs1], dims=("obs_id",))
        _ = pm.Deterministic("f_s_plot", f_s_all[i_plot0:i_plot1], dims=("plot_id",))
        _ = pm.Deterministic(
            "f_s_query", f_s_all[i_query0:i_query1], dims=("query_id",)
        )

        # ---- Shared standardised ages ----

        z_obs = pm.Deterministic(
            "z_obs", X_all_z_data[i_obs0:i_obs1, 0], dims=("obs_id",)
        )
        z_plot = pm.Deterministic(
            "z_plot", X_all_z_data[i_plot0:i_plot1, 0], dims=("plot_id",)
        )
        z_query = pm.Deterministic(
            "z_query", X_all_z_data[i_query0:i_query1, 0], dims=("query_id",)
        )

        # ============================================================
        # Kappa — understood
        # ============================================================

        kappa_min_u = config.kappa_min_u_dist.to_pymc("kappa_min_u")
        a_kappa_u = config.a_kappa_u_dist.to_pymc("a_kappa_u")
        b_kappa_mag_u = config.b_kappa_mag_u_dist.to_pymc("b_kappa_mag_u")
        b_kappa_u = pm.Deterministic("b_kappa_u", -b_kappa_mag_u)

        def kappa_u_of_z(z):
            return kappa_min_u + pm.math.exp(a_kappa_u + b_kappa_u * z)

        kappa_u_obs = pm.Deterministic(
            "kappa_u_obs", kappa_u_of_z(z_obs), dims="obs_id"
        )
        _ = pm.Deterministic("kappa_u_plot", kappa_u_of_z(z_plot), dims="plot_id")
        _ = pm.Deterministic("kappa_u_query", kappa_u_of_z(z_query), dims="query_id")

        # ============================================================
        # Kappa — spoken
        # ============================================================

        kappa_min_s = config.kappa_min_s_dist.to_pymc("kappa_min_s")
        a_kappa_s = config.a_kappa_s_dist.to_pymc("a_kappa_s")
        b_kappa_mag_s = config.b_kappa_mag_s_dist.to_pymc("b_kappa_mag_s")
        b_kappa_s = pm.Deterministic("b_kappa_s", -b_kappa_mag_s)

        def kappa_s_of_z(z):
            return kappa_min_s + pm.math.exp(a_kappa_s + b_kappa_s * z)

        kappa_s_obs = pm.Deterministic(
            "kappa_s_obs", kappa_s_of_z(z_obs), dims="obs_id"
        )
        _ = pm.Deterministic("kappa_s_plot", kappa_s_of_z(z_plot), dims="plot_id")
        _ = pm.Deterministic("kappa_s_query", kappa_s_of_z(z_query), dims="query_id")

        # ============================================================
        # Likelihoods — separate observation indices
        # ============================================================

        # Understood likelihood (only where observed)
        p_u_obs_sel = p_u_obs[idx_u]
        p_u_obs_clip = pm.math.clip(p_u_obs_sel, EPSILON, 1 - EPSILON)
        alpha_u = p_u_obs_clip * kappa_u_obs[idx_u]
        beta_u = (1 - p_u_obs_clip) * kappa_u_obs[idx_u]

        _ = pm.BetaBinomial(
            "y_u_obs",
            n=n_trials,
            alpha=alpha_u,
            beta=beta_u,
            observed=y_u_observed,
            dims=("obs_u_id",),
        )

        # Spoken likelihood (only where observed)
        p_s_obs_sel = p_s_obs[idx_s]
        p_s_obs_clip = pm.math.clip(p_s_obs_sel, EPSILON, 1 - EPSILON)
        alpha_s = p_s_obs_clip * kappa_s_obs[idx_s]
        beta_s = (1 - p_s_obs_clip) * kappa_s_obs[idx_s]

        _ = pm.BetaBinomial(
            "y_s_obs",
            n=n_trials,
            alpha=alpha_s,
            beta=beta_s,
            observed=y_s_observed,
            dims=("obs_s_id",),
        )

    variables = pymc_utils.get_variables_dict(model_pm)

    pymc_utils.report_model_summary(model_pm)

    digraph = pymc_utils.model_to_graphviz(model_pm)
    digraph.render(
        filename=os.path.join(context.reporting.output_dir, "gp_model_graph"),
        format="svg",
        cleanup=True,
    )

    context.set_model(model_pm, variables)


# ============================================================
# Sample extraction
# ============================================================


def _extract_posterior(trace, name, dim_name):
    """Extract posterior samples, stacking chains and draws."""
    return np.array(
        trace.posterior[name]
        .stack(sample=("chain", "draw"))
        .transpose(dim_name, "sample")
        .values
    )


def _extract_posterior_predictive(trace, name, dim_name):
    """Extract posterior predictive samples, stacking chains and draws."""
    return np.array(
        trace.posterior_predictive[name]
        .stack(sample=("chain", "draw"))
        .transpose(dim_name, "sample")
        .values,
        dtype=int,
    )


def extract_model_samples(trace: xr.DataTree) -> BivariateModelSamples:
    """Extract model samples into a structured format for plotting and reporting."""

    # Understood
    f_u_obs = _extract_posterior(trace, "f_u_obs", "obs_id")
    f_u_plot = _extract_posterior(trace, "f_u_plot", "plot_id")
    f_u_query = _extract_posterior(trace, "f_u_query", "query_id")

    p_u_obs = _extract_posterior(trace, "p_u_obs", "obs_id")
    p_u_plot = _extract_posterior(trace, "p_u_plot", "plot_id")
    p_u_query = _extract_posterior(trace, "p_u_query", "query_id")

    kappa_u_plot = _extract_posterior(trace, "kappa_u_plot", "plot_id")
    kappa_u_query = _extract_posterior(trace, "kappa_u_query", "query_id")

    # Production rate
    h_obs = _extract_posterior(trace, "h_obs", "obs_id")
    h_plot = _extract_posterior(trace, "h_plot", "plot_id")
    h_query = _extract_posterior(trace, "h_query", "query_id")

    q_obs = _extract_posterior(trace, "q_obs", "obs_id")
    q_plot = _extract_posterior(trace, "q_plot", "plot_id")
    q_query = _extract_posterior(trace, "q_query", "query_id")

    # Spoken (derived)
    f_s_obs = _extract_posterior(trace, "f_s_obs", "obs_id")
    f_s_plot = _extract_posterior(trace, "f_s_plot", "plot_id")
    f_s_query = _extract_posterior(trace, "f_s_query", "query_id")

    p_s_obs = _extract_posterior(trace, "p_s_obs", "obs_id")
    p_s_plot = _extract_posterior(trace, "p_s_plot", "plot_id")
    p_s_query = _extract_posterior(trace, "p_s_query", "query_id")

    kappa_s_plot = _extract_posterior(trace, "kappa_s_plot", "plot_id")
    kappa_s_query = _extract_posterior(trace, "kappa_s_query", "query_id")

    # Observed data — expand to full obs_id length with NaN where unobserved
    obs_u_mask = np.array(trace.constant_data["obs_u_mask"].values, dtype=bool)
    obs_s_mask = np.array(trace.constant_data["obs_s_mask"].values, dtype=bool)
    n_obs = len(obs_u_mask)

    y_u_obs_raw = np.array(trace.observed_data["y_u_obs"].values, dtype=float)
    y_u_obs = np.full(n_obs, np.nan)
    y_u_obs[obs_u_mask] = y_u_obs_raw

    y_s_obs_raw = np.array(trace.observed_data["y_s_obs"].values, dtype=float)
    y_s_obs = np.full(n_obs, np.nan)
    y_s_obs[obs_s_mask] = y_s_obs_raw

    # Posterior predictive
    y_u_plot = _extract_posterior_predictive(trace, "y_u_plot", "plot_id")
    y_u_query = _extract_posterior_predictive(trace, "y_u_query", "query_id")
    y_s_plot = _extract_posterior_predictive(trace, "y_s_plot", "plot_id")
    y_s_query = _extract_posterior_predictive(trace, "y_s_query", "query_id")

    # Constant data
    X_obs = np.array(trace.constant_data["X_obs"].values)
    X_plot = np.array(trace.constant_data["X_plot"].values)
    X_query = np.array(trace.constant_data["X_query"].values)

    # Standardised ages
    X_obs_z = _extract_posterior(trace, "z_obs", "obs_id")
    X_plot_z = _extract_posterior(trace, "z_plot", "plot_id")
    X_query_z = _extract_posterior(trace, "z_query", "query_id")

    return BivariateModelSamples(
        X_obs=X_obs,
        X_plot=X_plot,
        X_query=X_query,
        X_obs_z=X_obs_z,
        X_plot_z=X_plot_z,
        X_query_z=X_query_z,
        f_u_obs=f_u_obs,
        f_u_plot=f_u_plot,
        f_u_query=f_u_query,
        p_u_obs=p_u_obs,
        p_u_plot=p_u_plot,
        p_u_query=p_u_query,
        y_u_obs=y_u_obs,
        y_u_plot=y_u_plot,
        y_u_query=y_u_query,
        kappa_u_plot=kappa_u_plot,
        kappa_u_query=kappa_u_query,
        h_obs=h_obs,
        h_plot=h_plot,
        h_query=h_query,
        q_obs=q_obs,
        q_plot=q_plot,
        q_query=q_query,
        f_s_obs=f_s_obs,
        f_s_plot=f_s_plot,
        f_s_query=f_s_query,
        p_s_obs=p_s_obs,
        p_s_plot=p_s_plot,
        p_s_query=p_s_query,
        y_s_obs=y_s_obs,
        y_s_plot=y_s_plot,
        y_s_query=y_s_query,
        kappa_s_plot=kappa_s_plot,
        kappa_s_query=kappa_s_query,
        obs_u_mask=obs_u_mask,
        obs_s_mask=obs_s_mask,
    )


# ============================================================
# Pipeline steps
# ============================================================


def prior_predictive_checks(context: BivariateContext):
    """Run prior predictive checks."""
    with context.model:
        prior_samples = pm.sample_prior_predictive(
            draws=1000,
            random_seed=context.sampling.random_seed,
            compile_kwargs=dict(mode="FAST_COMPILE")
        )

    context.set_prior_samples(prior_samples)

    analysis_df = context.analysis_df
    has_u = analysis_df["understood"].notna()
    has_s = analysis_df["spoken"].notna()

    # Prior samples for understood trajectory
    p_u_plot_samples = (
        prior_samples.prior["p_u_plot"]
        .stack(sample=("chain", "draw"))
        .transpose("plot_id", "sample")
    )

    plotting.plot_prior_samples(
        prior_samples.constant_data["X_plot"].values,
        p_u_plot_samples.values,
        analysis_df.loc[has_u, "age"],
        analysis_df.loc[has_u, "understood"],
        n_trials=context.model_data.n_trials,
        n_curves=1000,
        x_label="Age (months)",
        y_label="Words understood",
        filename="prior_samples_u",
        output_dir=context.reporting.output_dir,
    )

    # Prior samples for spoken trajectory
    p_s_plot_samples = (
        prior_samples.prior["p_s_plot"]
        .stack(sample=("chain", "draw"))
        .transpose("plot_id", "sample")
    )

    plotting.plot_prior_samples(
        prior_samples.constant_data["X_plot"].values,
        p_s_plot_samples.values,
        analysis_df.loc[has_s, "age"],
        analysis_df.loc[has_s, "spoken"],
        n_trials=context.model_data.n_trials,
        n_curves=1000,
        x_label="Age (months)",
        y_label="Words spoken",
        filename="prior_samples_s",
        output_dir=context.reporting.output_dir,
    )

    # Prior samples for production rate q(a)
    q_plot_samples = (
        prior_samples.prior["q_plot"]
        .stack(sample=("chain", "draw"))
        .transpose("plot_id", "sample")
    )

    fig, ax = plt.subplots(figsize=plot_styles.FIGSIZE_XL)
    X_plot_vals = prior_samples.constant_data["X_plot"].values
    n_curves = min(500, q_plot_samples.shape[1])
    for i in range(n_curves):
        ax.plot(X_plot_vals, q_plot_samples.values[:, i], alpha=0.01)
    ax.set_xlabel("Age (months)")
    ax.set_ylabel("q(a) = p_S(a) / p_U(a)")
    ax.set_ylim(0, 1)
    fig.savefig(
        os.path.join(context.reporting.output_dir, "prior_samples_q.png"), dpi=300
    )
    fig.savefig(os.path.join(context.reporting.output_dir, "prior_samples_q.svg"))
    context.plots["prior_samples_q"] = fig
    plt.close()


def sample(context: BivariateContext):
    """Draw samples from the posterior using MCMC."""
    config_table("Sampling configuration", context.sampling)

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


def diagnostics(context: BivariateContext):
    """Run diagnostics on the posterior samples."""
    var_names = [
        var.name for var in context.model.unobserved_RVs if var.size.eval() <= 2
    ]
    diagnostics_df = az.summary(
        context.trace,
        var_names=var_names,
        round_to=3,
        ci_prob=context.reporting.hdi,
        ci_kind="hdi",
    )

    diagnostics_df.to_csv(
        os.path.join(context.reporting.output_dir, "diagnostics.csv"), index=True
    )

    dataframe_table(diagnostics_df, title="Posterior diagnostics")
    _report_diagnostic_warnings(diagnostics_df)

    # KDE pair plot
    with az.rc_context(pair_plot_rc_params(context.trace, var_names)):
        plot_diagnostics_mcmc.plot_kde_pair(
            context.trace,
            var_names=var_names,
            output_dir=context.reporting.output_dir,
            filename="pair_plot",
        )
    context.plots["pair_plot"] = plt.gcf()
    plt.close()

    # Trace plot
    var_names_ext = var_names + ["kappa_u_obs", "kappa_s_obs"]

    az.plot_trace(
        context.trace,
        combined=True,
        var_names=var_names_ext,
    )
    plt.savefig(os.path.join(context.reporting.output_dir, "trace_plot.png"), dpi=300)
    context.plots["trace_plot"] = plt.gcf()
    plt.close()

    # Energy plot
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

    # LOO-CV
    with context.model:
        trace = pm.compute_log_likelihood(context.trace)

    context.set_trace(trace)

    loocv_s = az.loo(context.trace, var_name="y_s_obs")
    loocv_u = az.loo(context.trace, var_name="y_u_obs")
    context.set_loocv({"y_s_obs": loocv_s, "y_u_obs": loocv_u})
    heading("LOO-CV — words spoken", style="bold cyan")
    console.print(loocv_s)
    heading("LOO-CV — words understood", style="bold cyan")
    console.print(loocv_u)


def sample_posterior_predictive(context: BivariateContext, definition=None):
    """Sample from the posterior predictive distribution.

    For models with subject-level random intercepts (use_subject_re_u and/or
    use_subject_re_q), the y_*_plot / y_*_query nodes are constructed using
    marginal probabilities that integrate over a freshly-sampled subject RE
    drawn from the corresponding prior. This makes y_u_query etc. the
    correct predictive distribution for an unseen DS child, rather than
    the population-mean conditional distribution.
    """
    n_trials = context.model_data.n_trials

    p_u_plot = context.model_variables["p_u_plot"]
    p_u_query = context.model_variables["p_u_query"]
    kappa_u_plot = context.model_variables["kappa_u_plot"]
    kappa_u_query = context.model_variables["kappa_u_query"]

    p_s_plot = context.model_variables["p_s_plot"]
    p_s_query = context.model_variables["p_s_query"]
    kappa_s_plot = context.model_variables["kappa_s_plot"]
    kappa_s_query = context.model_variables["kappa_s_query"]

    use_subject_re_u = bool(getattr(definition, "use_subject_re_u", False))
    use_subject_re_q = bool(getattr(definition, "use_subject_re_q", False))

    with context.model:
        # Subject-marginalised probabilities if subject REs are present.
        # delta_subj_*_marg_{plot,query} are auxiliary RVs sampled from the
        # subject-RE prior during sample_posterior_predictive — they are not
        # part of the likelihood and do not affect posterior sampling.
        if use_subject_re_u:
            tau_subj_u = context.model_variables["tau_subj_u"]
            f_u_plot_var = context.model_variables["f_u_plot"]
            f_u_query_var = context.model_variables["f_u_query"]
            delta_u_plot_marg = pm.Normal(
                "_delta_subj_u_plot_marg", mu=0.0, sigma=tau_subj_u, dims="plot_id"
            )
            delta_u_query_marg = pm.Normal(
                "_delta_subj_u_query_marg", mu=0.0, sigma=tau_subj_u, dims="query_id"
            )
            p_u_plot = pm.math.sigmoid(f_u_plot_var + delta_u_plot_marg)
            p_u_query = pm.math.sigmoid(f_u_query_var + delta_u_query_marg)

        if use_subject_re_q:
            tau_subj_q = context.model_variables["tau_subj_q"]
            h_plot_var = context.model_variables["h_plot"]
            h_query_var = context.model_variables["h_query"]
            delta_q_plot_marg = pm.Normal(
                "_delta_subj_q_plot_marg", mu=0.0, sigma=tau_subj_q, dims="plot_id"
            )
            delta_q_query_marg = pm.Normal(
                "_delta_subj_q_query_marg", mu=0.0, sigma=tau_subj_q, dims="query_id"
            )
            q_plot_marg = pm.math.sigmoid(h_plot_var + delta_q_plot_marg)
            q_query_marg = pm.math.sigmoid(h_query_var + delta_q_query_marg)
            p_s_plot = p_u_plot * q_plot_marg
            p_s_query = p_u_query * q_query_marg
        elif use_subject_re_u:
            # Marginal U but not q: rebuild p_s from new p_u and original q.
            q_plot = context.model_variables["q_plot"]
            q_query = context.model_variables["q_query"]
            p_s_plot = p_u_plot * q_plot
            p_s_query = p_u_query * q_query

        # Understood — plot
        p_u_plot_clip = pm.math.clip(p_u_plot, EPSILON, 1 - EPSILON)
        pm.BetaBinomial(
            "y_u_plot",
            n=n_trials,
            alpha=p_u_plot_clip * kappa_u_plot,
            beta=(1 - p_u_plot_clip) * kappa_u_plot,
            dims=("plot_id",),
        )
        # Understood — query
        p_u_query_clip = pm.math.clip(p_u_query, EPSILON, 1 - EPSILON)
        pm.BetaBinomial(
            "y_u_query",
            n=n_trials,
            alpha=p_u_query_clip * kappa_u_query,
            beta=(1 - p_u_query_clip) * kappa_u_query,
            dims=("query_id",),
        )
        # Spoken — plot
        p_s_plot_clip = pm.math.clip(p_s_plot, EPSILON, 1 - EPSILON)
        pm.BetaBinomial(
            "y_s_plot",
            n=n_trials,
            alpha=p_s_plot_clip * kappa_s_plot,
            beta=(1 - p_s_plot_clip) * kappa_s_plot,
            dims=("plot_id",),
        )
        # Spoken — query
        p_s_query_clip = pm.math.clip(p_s_query, EPSILON, 1 - EPSILON)
        pm.BetaBinomial(
            "y_s_query",
            n=n_trials,
            alpha=p_s_query_clip * kappa_s_query,
            beta=(1 - p_s_query_clip) * kappa_s_query,
            dims=("query_id",),
        )

        trace = pm.sample_posterior_predictive(
            context.trace,
            var_names=[
                "y_u_plot",
                "y_u_query",
                "y_u_obs",
                "y_s_plot",
                "y_s_query",
                "y_s_obs",
            ],
            extend_inferencedata=True,
            random_seed=context.sampling.random_seed,
        )

    context.set_trace(trace)

    trace.to_netcdf(os.path.join(context.reporting.output_dir, "trace.nc"))

    sample_data = extract_model_samples(context.trace)
    context.set_model_samples(sample_data)


def posterior_summary(context: BivariateContext):
    """Compute and store the posterior summary tables at query ages."""
    samples = context.model_samples
    n_trials = context.model_data.n_trials
    hdi_prob = context.reporting.hdi

    # Understood summary
    summary_u = posterior_analysis.posterior_summary_table(
        samples.X_query,
        samples.p_u_query,
        samples.y_u_query,
        n_trials=n_trials,
        hdi_prob=hdi_prob,
    )
    dataframe_table(
        summary_u, title="Posterior summary — words understood", show_index=False
    )
    context.dataframes["posterior_summary_u"] = summary_u
    summary_u.to_csv(
        os.path.join(context.reporting.output_dir, "posterior_summary_u.csv"),
        index=False,
    )

    # Spoken summary
    summary_s = posterior_analysis.posterior_summary_table(
        samples.X_query,
        samples.p_s_query,
        samples.y_s_query,
        n_trials=n_trials,
        hdi_prob=hdi_prob,
    )
    dataframe_table(
        summary_s, title="Posterior summary — words spoken", show_index=False
    )
    context.dataframes["posterior_summary_s"] = summary_s
    summary_s.to_csv(
        os.path.join(context.reporting.output_dir, "posterior_summary_s.csv"),
        index=False,
    )

    # Production rate summary
    q_query_median = np.median(samples.q_query, axis=1)
    q_query_hdi = az.hdi(samples.q_query.T, prob=hdi_prob)

    summary_q = pd.DataFrame(
        {
            "age_months": samples.X_query,
            "q_median": q_query_median,
            "q_hdi_lo": q_query_hdi[:, 0],
            "q_hdi_hi": q_query_hdi[:, 1],
        }
    )
    dataframe_table(
        summary_q, title="Posterior summary — production rate q(a)", show_index=False
    )
    context.dataframes["posterior_summary_q"] = summary_q
    summary_q.to_csv(
        os.path.join(context.reporting.output_dir, "posterior_summary_q.csv"),
        index=False,
    )


# ============================================================
# Bivariate-specific plotting functions
# ============================================================


def plot_understood_spoken_trajectory(
    samples: BivariateModelSamples,
    n_trials: int,
    output_dir: str | None = None,
    filename: str | None = None,
):
    """Plot both understood and spoken posterior predictive median trends on one figure."""
    X_plot = samples.X_plot

    # Understood
    y_u_median = np.quantile(samples.y_u_plot, 0.50, axis=1)
    y_u_90 = np.quantile(samples.y_u_plot, [0.05, 0.95], axis=1).T
    y_u_50 = np.quantile(samples.y_u_plot, [0.25, 0.75], axis=1).T

    # Spoken
    y_s_median = np.quantile(samples.y_s_plot, 0.50, axis=1)
    y_s_90 = np.quantile(samples.y_s_plot, [0.05, 0.95], axis=1).T
    y_s_50 = np.quantile(samples.y_s_plot, [0.25, 0.75], axis=1).T

    fig, ax = plt.subplots(figsize=plot_styles.FIGSIZE_XL)

    # Understood bands
    ax.fill_between(X_plot, y_u_90[:, 0], y_u_90[:, 1], alpha=0.15, color="C0")
    ax.fill_between(X_plot, y_u_50[:, 0], y_u_50[:, 1], alpha=0.25, color="C0")
    ax.plot(X_plot, y_u_median, lw=3, color="C0", label="Words understood (median)")

    # Spoken bands
    ax.fill_between(X_plot, y_s_90[:, 0], y_s_90[:, 1], alpha=0.15, color="C1")
    ax.fill_between(X_plot, y_s_50[:, 0], y_s_50[:, 1], alpha=0.25, color="C1")
    ax.plot(X_plot, y_s_median, lw=3, color="C1", label="Words spoken (median)")

    # Observed data
    X_obs = samples.X_obs
    u_mask = ~np.isnan(samples.y_u_obs)
    if u_mask.any():
        ax.scatter(X_obs[u_mask], samples.y_u_obs[u_mask], s=10, alpha=0.2, color="C0")
    s_mask = ~np.isnan(samples.y_s_obs)
    if s_mask.any():
        ax.scatter(X_obs[s_mask], samples.y_s_obs[s_mask], s=10, alpha=0.2, color="C1")

    ax.set_xlabel("Age (months)")
    ax.set_ylabel("Word count")
    ax.legend(loc="upper left", frameon=True)
    ax.set_ylim(-20, n_trials + 50)

    if output_dir is not None and filename is not None:
        fig.savefig(os.path.join(output_dir, f"{filename}.png"), dpi=300)
        fig.savefig(os.path.join(output_dir, f"{filename}.svg"))
        _save_csv(pd.DataFrame({
            "age_months": X_plot,
            "understood_median": y_u_median,
            "understood_p05": y_u_90[:, 0],
            "understood_p95": y_u_90[:, 1],
            "understood_p25": y_u_50[:, 0],
            "understood_p75": y_u_50[:, 1],
            "spoken_median": y_s_median,
            "spoken_p05": y_s_90[:, 0],
            "spoken_p95": y_s_90[:, 1],
            "spoken_p25": y_s_50[:, 0],
            "spoken_p75": y_s_50[:, 1],
        }), output_dir, filename)

    return fig


def plot_understood_spoken_trajectory_hdi(
    samples: BivariateModelSamples,
    n_trials: int,
    output_dir: str | None = None,
    filename: str | None = None,
):
    """Plot understood and spoken posterior predictive medians with 50% and 75% HDI bands."""
    X_plot = samples.X_plot

    # Understood HDI bands
    y_u_median = np.median(samples.y_u_plot, axis=1)
    y_u_hdi_75 = az.hdi(samples.y_u_plot.T, prob=0.75)
    y_u_hdi_50 = az.hdi(samples.y_u_plot.T, prob=0.50)

    # Spoken HDI bands
    y_s_median = np.median(samples.y_s_plot, axis=1)
    y_s_hdi_75 = az.hdi(samples.y_s_plot.T, prob=0.75)
    y_s_hdi_50 = az.hdi(samples.y_s_plot.T, prob=0.50)

    fig, ax = plt.subplots(figsize=plot_styles.FIGSIZE_XL)

    # Understood bands
    ax.fill_between(X_plot, y_u_hdi_75[:, 0], y_u_hdi_75[:, 1], alpha=0.15, color="C0", label="Words understood (75% HDI)")
    ax.fill_between(X_plot, y_u_hdi_50[:, 0], y_u_hdi_50[:, 1], alpha=0.25, color="C0", label="Words understood (50% HDI)")
    ax.plot(X_plot, y_u_median, lw=3, color="C0", label="Words understood (median)")

    # Spoken bands
    ax.fill_between(X_plot, y_s_hdi_75[:, 0], y_s_hdi_75[:, 1], alpha=0.15, color="C1", label="Words spoken (75% HDI)")
    ax.fill_between(X_plot, y_s_hdi_50[:, 0], y_s_hdi_50[:, 1], alpha=0.25, color="C1", label="Words spoken (50% HDI)")
    ax.plot(X_plot, y_s_median, lw=3, color="C1", label="Words spoken (median)")

    ax.set_xlabel("Age (months)")
    ax.set_ylabel("Word count")
    ax.legend(loc="upper left", frameon=True)
    ax.set_ylim(-20, n_trials + 50)

    if output_dir is not None and filename is not None:
        fig.savefig(os.path.join(output_dir, f"{filename}.png"), dpi=300)
        fig.savefig(os.path.join(output_dir, f"{filename}.svg"))
        _save_csv(pd.DataFrame({
            "age_months": X_plot,
            "understood_median": y_u_median,
            "understood_hdi75_lo": y_u_hdi_75[:, 0],
            "understood_hdi75_hi": y_u_hdi_75[:, 1],
            "understood_hdi50_lo": y_u_hdi_50[:, 0],
            "understood_hdi50_hi": y_u_hdi_50[:, 1],
            "spoken_median": y_s_median,
            "spoken_hdi75_lo": y_s_hdi_75[:, 0],
            "spoken_hdi75_hi": y_s_hdi_75[:, 1],
            "spoken_hdi50_lo": y_s_hdi_50[:, 0],
            "spoken_hdi50_hi": y_s_hdi_50[:, 1],
        }), output_dir, filename)

    return fig


def plot_production_rate(
    samples: BivariateModelSamples,
    hdi_prob: float = 0.90,
    output_dir: str | None = None,
    filename: str | None = None,
):
    """Plot the posterior of the production ratio q(a) = p_S(a) / p_U(a) over age."""
    X_plot = samples.X_plot
    q_plot = samples.q_plot

    q_median = np.median(q_plot, axis=1)
    q_hdi = az.hdi(q_plot.T, prob=hdi_prob)
    q_hdi_75 = az.hdi(q_plot.T, prob=0.75)
    q_hdi_50 = az.hdi(q_plot.T, prob=0.50)

    fig, ax = plt.subplots(figsize=plot_styles.FIGSIZE_XL)

    ax.fill_between(
        X_plot,
        q_hdi[:, 0],
        q_hdi[:, 1],
        alpha=0.20,
        label=f"{int(hdi_prob * 100)}% HDI",
    )
    ax.fill_between(
        X_plot,
        q_hdi_75[:, 0],
        q_hdi_75[:, 1],
        alpha=0.25,
        label="75% HDI",
    )
    ax.fill_between(
        X_plot,
        q_hdi_50[:, 0],
        q_hdi_50[:, 1],
        alpha=0.30,
        label="50% HDI",
    )
    ax.plot(X_plot, q_median, lw=3, label="Median q(a)")

    ax.set_xlabel("Age (months)")
    ax.set_ylabel("q(a) = p_S(a) / p_U(a)")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left", frameon=True)
    ax.set_title("Production ratio q(a)")

    if output_dir is not None and filename is not None:
        fig.savefig(os.path.join(output_dir, f"{filename}.png"), dpi=300)
        fig.savefig(os.path.join(output_dir, f"{filename}.svg"))
        _save_csv(pd.DataFrame({
            "age_months": X_plot,
            "q_median": q_median,
            "hdi_lo": q_hdi[:, 0],
            "hdi_hi": q_hdi[:, 1],
            "hdi75_lo": q_hdi_75[:, 0],
            "hdi75_hi": q_hdi_75[:, 1],
            "hdi50_lo": q_hdi_50[:, 0],
            "hdi50_hi": q_hdi_50[:, 1],
        }), output_dir, filename)

    return fig


def plot_production_rate_by_understood(
    samples: BivariateModelSamples,
    n_trials: int,
    hdi_prob: float = 0.90,
    output_dir: str | None = None,
    filename: str | None = None,
):
    """Plot production ratio q against expected words understood (median p_U * n_trials)."""
    p_u_plot = samples.p_u_plot  # (n_plot, n_samples)
    q_plot = samples.q_plot  # (n_plot, n_samples)

    x_words = np.median(p_u_plot, axis=1) * n_trials
    q_median = np.median(q_plot, axis=1)
    q_hdi = az.hdi(q_plot.T, prob=hdi_prob)
    q_hdi_75 = az.hdi(q_plot.T, prob=0.75)
    q_hdi_50 = az.hdi(q_plot.T, prob=0.50)

    fig, ax = plt.subplots(figsize=plot_styles.FIGSIZE_XL)

    ax.fill_between(
        x_words,
        q_hdi[:, 0],
        q_hdi[:, 1],
        alpha=0.20,
        label=f"{int(hdi_prob * 100)}% HDI",
    )
    ax.fill_between(
        x_words,
        q_hdi_75[:, 0],
        q_hdi_75[:, 1],
        alpha=0.25,
        label="75% HDI",
    )
    ax.fill_between(
        x_words,
        q_hdi_50[:, 0],
        q_hdi_50[:, 1],
        alpha=0.30,
        label="50% HDI",
    )
    ax.plot(x_words, q_median, lw=3, label="Median q")

    ax.set_xlabel("Expected words understood")
    ax.set_ylabel("q = p_S / p_U")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left", frameon=True)
    ax.set_title("Production ratio by words understood")

    if output_dir is not None and filename is not None:
        fig.savefig(os.path.join(output_dir, f"{filename}.png"), dpi=300)
        fig.savefig(os.path.join(output_dir, f"{filename}.svg"))
        _save_csv(pd.DataFrame({
            "words_understood": x_words,
            "q_median": q_median,
            "hdi_lo": q_hdi[:, 0],
            "hdi_hi": q_hdi[:, 1],
            "hdi75_lo": q_hdi_75[:, 0],
            "hdi75_hi": q_hdi_75[:, 1],
            "hdi50_lo": q_hdi_50[:, 0],
            "hdi50_hi": q_hdi_50[:, 1],
        }), output_dir, filename)

    return fig


def plot_production_rate_predictive(
    samples: BivariateModelSamples,
    output_dir: str | None = None,
    filename: str | None = None,
):
    """Plot posterior predictive spoken/understood count ratio with 50% and 75% HDI bands.

    Uses posterior predictive counts (y_s / y_u). Samples where y_u == 0 are
    excluded per age point before computing summary statistics.
    """
    X_plot = samples.X_plot
    y_u = samples.y_u_plot  # (n_plot, n_samples)
    y_s = samples.y_s_plot

    n_ages = y_u.shape[0]
    ratio_median = np.empty(n_ages)
    ratio_hdi_75 = np.empty((n_ages, 2))
    ratio_hdi_50 = np.empty((n_ages, 2))

    for i in range(n_ages):
        mask = y_u[i] > 0
        if mask.sum() < 10:
            ratio_median[i] = np.nan
            ratio_hdi_75[i] = np.nan
            ratio_hdi_50[i] = np.nan
            continue
        ratio_i = y_s[i, mask] / y_u[i, mask]
        ratio_median[i] = np.median(ratio_i)
        ratio_hdi_75[i] = az.hdi(ratio_i, prob=0.75)
        ratio_hdi_50[i] = az.hdi(ratio_i, prob=0.50)

    fig, ax = plt.subplots(figsize=plot_styles.FIGSIZE_XL)

    ax.fill_between(
        X_plot,
        ratio_hdi_75[:, 0],
        ratio_hdi_75[:, 1],
        alpha=0.20,
        label="75% HDI",
    )
    ax.fill_between(
        X_plot,
        ratio_hdi_50[:, 0],
        ratio_hdi_50[:, 1],
        alpha=0.30,
        label="50% HDI",
    )
    ax.plot(X_plot, ratio_median, lw=3, label="Median")

    ax.set_xlabel("Age (months)")
    ax.set_ylabel("Words spoken / words understood")
    ax.set_ylim(0, 1.05)
    ax.axhline(1.0, ls=":", lw=1, color="grey", alpha=0.5)
    ax.legend(loc="upper left", frameon=True)
    ax.set_title("Posterior predictive production ratio")

    if output_dir is not None and filename is not None:
        fig.savefig(os.path.join(output_dir, f"{filename}.png"), dpi=300)
        fig.savefig(os.path.join(output_dir, f"{filename}.svg"))
        _save_csv(pd.DataFrame({
            "age_months": X_plot,
            "ratio_median": ratio_median,
            "hdi75_lo": ratio_hdi_75[:, 0],
            "hdi75_hi": ratio_hdi_75[:, 1],
            "hdi50_lo": ratio_hdi_50[:, 0],
            "hdi50_hi": ratio_hdi_50[:, 1],
        }), output_dir, filename)

    return fig


def plot_comprehension_production_gap(
    samples: BivariateModelSamples,
    n_trials: int,
    hdi_prob: float = 0.90,
    output_dir: str | None = None,
    filename: str | None = None,
):
    """Plot the posterior of the comprehension-production gap (p_U - p_S) over age."""
    X_plot = samples.X_plot
    gap = (samples.p_u_plot - samples.p_s_plot) * n_trials  # in word count units

    gap_median = np.median(gap, axis=1)
    gap_hdi = az.hdi(gap.T, prob=hdi_prob)
    gap_hdi_50 = az.hdi(gap.T, prob=0.50)

    fig, ax = plt.subplots(figsize=plot_styles.FIGSIZE_XL)

    ax.fill_between(
        X_plot,
        gap_hdi[:, 0],
        gap_hdi[:, 1],
        alpha=0.20,
        label=f"{int(hdi_prob * 100)}% HDI",
    )
    ax.fill_between(
        X_plot,
        gap_hdi_50[:, 0],
        gap_hdi_50[:, 1],
        alpha=0.30,
        label="50% HDI",
    )
    ax.plot(X_plot, gap_median, lw=3, label="Median gap")

    ax.set_xlabel("Age (months)")
    ax.set_ylabel("E[understood] - E[spoken] (words)")
    ax.legend(loc="upper left", frameon=True)
    ax.set_title("Comprehension-production gap")

    if output_dir is not None and filename is not None:
        fig.savefig(os.path.join(output_dir, f"{filename}.png"), dpi=300)
        fig.savefig(os.path.join(output_dir, f"{filename}.svg"))
        _save_csv(pd.DataFrame({
            "age_months": X_plot,
            "gap_median": gap_median,
            "hdi_lo": gap_hdi[:, 0],
            "hdi_hi": gap_hdi[:, 1],
            "hdi50_lo": gap_hdi_50[:, 0],
            "hdi50_hi": gap_hdi_50[:, 1],
        }), output_dir, filename)

    return fig


def plot_understood_vs_spoken(
    samples: BivariateModelSamples,
    n_trials: int,
    n_draws: int = 200,
    output_dir: str | None = None,
    filename: str | None = None,
):
    """Plot posterior expected words understood (x) vs words spoken (y) as spaghetti curves."""
    E_u = samples.p_u_plot * n_trials  # (n_plot, n_samples)
    E_s = samples.p_s_plot * n_trials

    n_available = E_u.shape[1]
    n_draws = min(n_draws, n_available)
    idx = np.random.default_rng(42).choice(n_available, size=n_draws, replace=False)

    E_u_median = np.median(E_u, axis=1)
    E_s_median = np.median(E_s, axis=1)

    fig, ax = plt.subplots(figsize=plot_styles.FIGSIZE_XL)

    for i in idx:
        ax.plot(E_u[:, i], E_s[:, i], lw=0.3, alpha=0.15, color="C0")

    ax.plot(E_u_median, E_s_median, lw=3, color="C0", label="Median")

    # Reference line: understood = spoken
    limit = max(E_u_median.max(), E_s_median.max()) * 1.05
    ax.plot([0, limit], [0, limit], ls="--", lw=1, color="grey", label="y = x")

    ax.set_xlabel("E[words understood]")
    ax.set_ylabel("E[words spoken]")
    ax.set_title("Expected words understood vs spoken")
    ax.legend(loc="upper left", frameon=True)

    if output_dir is not None and filename is not None:
        fig.savefig(os.path.join(output_dir, f"{filename}.png"), dpi=300)
        fig.savefig(os.path.join(output_dir, f"{filename}.svg"))
        _save_csv(pd.DataFrame({
            "age_months": samples.X_plot,
            "understood_median": E_u_median,
            "spoken_median": E_s_median,
        }), output_dir, filename)

    return fig


def plot_understood_vs_spoken_predictive(
    samples: BivariateModelSamples,
    n_trials: int,
    n_draws: int = 200,
    output_dir: str | None = None,
    filename: str | None = None,
):
    """Plot posterior predictive words understood (x) vs words spoken (y) as spaghetti curves."""
    y_u = samples.y_u_plot  # (n_plot, n_samples)
    y_s = samples.y_s_plot

    n_available = y_u.shape[1]
    n_draws = min(n_draws, n_available)
    idx = np.random.default_rng(42).choice(n_available, size=n_draws, replace=False)

    y_u_median = np.median(y_u, axis=1)
    y_s_median = np.median(y_s, axis=1)

    fig, ax = plt.subplots(figsize=plot_styles.FIGSIZE_XL)

    for i in idx:
        ax.plot(y_u[:, i], y_s[:, i], lw=0.3, alpha=0.10, color="C0")

    ax.plot(y_u_median, y_s_median, lw=3, color="C0", label="Median")

    # Reference line: understood = spoken
    limit = max(y_u_median.max(), y_s_median.max()) * 1.05
    ax.plot([0, limit], [0, limit], ls="--", lw=1, color="grey", label="y = x")

    ax.set_xlabel("Words understood")
    ax.set_ylabel("Words spoken")
    ax.set_title("Posterior predictive words understood vs spoken")
    ax.legend(loc="upper left", frameon=True)

    if output_dir is not None and filename is not None:
        fig.savefig(os.path.join(output_dir, f"{filename}.png"), dpi=300)
        fig.savefig(os.path.join(output_dir, f"{filename}.svg"))
        _save_csv(pd.DataFrame({
            "age_months": samples.X_plot,
            "understood_median": y_u_median,
            "spoken_median": y_s_median,
        }), output_dir, filename)

    return fig


# ============================================================
# Shared bivariate plotting pipeline
# ============================================================


def _run_bivariate_outcome_plots(
    samples: BivariateModelSamples,
    y_plot: np.ndarray,
    y_query: np.ndarray,
    f_plot: np.ndarray,
    kappa_plot: np.ndarray,
    kappa_query: np.ndarray,
    x_obs: pd.Series,
    y_obs: pd.Series,
    n_trials: int,
    hdi_prob: float,
    output_dir: str,
    suffix: str,
    outcome_label: str,
    y_label: str,
):
    """Run the standard per-outcome plotting pipeline for a bivariate model."""
    plotting.plot_posterior_predictive_count_distributions_by_query_age(
        X_query=samples.X_query,
        y_query=y_query,
        n_trials=n_trials,
        output_dir=output_dir,
        filename=f"posterior_predictive_count_distributions_{suffix}",
        x_label=f"{outcome_label} (count)",
    )

    plotting.plot_posterior_predictive_pmf(
        samples.X_query,
        samples.X_plot,
        y_plot,
        n_trials,
        output_dir=output_dir,
        filename=f"posterior_predictive_pmf_{suffix}",
        x_label=f"{outcome_label} (count)",
    )

    plotting.plot_posterior_predictive_cdf(
        samples.X_query,
        samples.X_plot,
        y_plot,
        n_trials,
        output_dir=output_dir,
        filename=f"posterior_predictive_cdf_{suffix}",
        x_label=f"{outcome_label} (count)",
    )

    plotting.plot_posterior_predictive_median_trend(
        samples.X_plot,
        y_plot,
        x_obs,
        y_obs,
        output_dir=output_dir,
        filename=f"posterior_predictive_median_trend_{suffix}",
        y_label=y_label,
    )

    plotting.plot_posterior_predictive_median_trend(
        samples.X_plot,
        y_plot,
        x_obs,
        y_obs,
        smooth=True,
        savgol_window_length=15,
        savgol_polyorder=3,
        smooth_intervals=True,
        output_dir=output_dir,
        filename=f"posterior_predictive_median_trend_{suffix}_smoothed",
        y_label=y_label,
    )

    plotting.plot_expected_learning_rate(
        samples.X_plot,
        f_plot,
        n_trials=n_trials,
        hdi_prob=hdi_prob,
        output_dir=output_dir,
        filename=f"expected_learning_rate_{suffix}",
        y_label=f"Estimated {outcome_label.lower()} gain per month",
    )

    plotting.plot_expected_learning_rate(
        samples.X_plot,
        f_plot,
        n_trials=n_trials,
        hdi_prob=hdi_prob,
        smooth=True,
        savgol_window_length=15,
        savgol_polyorder=3,
        smooth_intervals=True,
        output_dir=output_dir,
        filename=f"expected_learning_rate_{suffix}_smoothed",
        y_label=f"Estimated {outcome_label.lower()} gain per month",
    )

    plotting.plot_posterior_kappa(
        samples.X_plot,
        kappa_plot,
        samples.X_query,
        kappa_query,
        n_trials=n_trials,
        hdi_prob=hdi_prob,
        output_dir=output_dir,
        filename=f"posterior_kappa_{suffix}",
    )


def _run_bivariate_joint_plots(
    context: BivariateContext,
    definition: BivariateModelDefinition,
):
    """Run the joint bivariate plots and per-outcome plots shared by VG05–VG07."""
    samples = context.model_samples
    analysis_df = context.analysis_df
    has_u = analysis_df["understood"].notna()
    has_s = analysis_df["spoken"].notna()

    # ---- Joint trajectory plot ----

    fig = plot_understood_spoken_trajectory(
        samples,
        n_trials=context.model_data.n_trials,
        output_dir=context.reporting.output_dir,
        filename="joint_trajectory",
    )
    context.plots["joint_trajectory"] = fig
    plt.close(fig)

    # ---- Joint trajectory HDI plot ----

    fig = plot_understood_spoken_trajectory_hdi(
        samples,
        n_trials=context.model_data.n_trials,
        output_dir=context.reporting.output_dir,
        filename="joint_trajectory_hdi",
    )
    context.plots["joint_trajectory_hdi"] = fig
    plt.close(fig)

    # ---- Production rate q(a) ----

    fig = plot_production_rate(
        samples,
        hdi_prob=context.reporting.hdi,
        output_dir=context.reporting.output_dir,
        filename="production_rate",
    )
    context.plots["production_rate"] = fig
    plt.close(fig)

    # ---- Production rate by words understood ----

    fig = plot_production_rate_by_understood(
        samples,
        n_trials=definition.n_trials,
        hdi_prob=context.reporting.hdi,
        output_dir=context.reporting.output_dir,
        filename="production_rate_by_understood",
    )
    context.plots["production_rate_by_understood"] = fig
    plt.close(fig)

    # ---- Posterior predictive production rate ----

    fig = plot_production_rate_predictive(
        samples,
        output_dir=context.reporting.output_dir,
        filename="production_rate_predictive",
    )
    context.plots["production_rate_predictive"] = fig
    plt.close(fig)

    # ---- Comprehension-production gap ----

    fig = plot_comprehension_production_gap(
        samples,
        n_trials=context.model_data.n_trials,
        hdi_prob=context.reporting.hdi,
        output_dir=context.reporting.output_dir,
        filename="comprehension_production_gap",
    )
    context.plots["comprehension_production_gap"] = fig
    plt.close(fig)

    # ---- Understood vs spoken ----

    fig = plot_understood_vs_spoken(
        samples,
        n_trials=context.model_data.n_trials,
        output_dir=context.reporting.output_dir,
        filename="understood_vs_spoken",
    )
    context.plots["understood_vs_spoken"] = fig
    plt.close(fig)

    # ---- Posterior predictive understood vs spoken ----

    fig = plot_understood_vs_spoken_predictive(
        samples,
        n_trials=context.model_data.n_trials,
        output_dir=context.reporting.output_dir,
        filename="understood_vs_spoken_predictive",
    )
    context.plots["understood_vs_spoken_predictive"] = fig
    plt.close(fig)

    # ---- Per-outcome plots: understood ----

    _run_bivariate_outcome_plots(
        samples=samples,
        y_plot=samples.y_u_plot,
        y_query=samples.y_u_query,
        f_plot=samples.f_u_plot,
        kappa_plot=samples.kappa_u_plot,
        kappa_query=samples.kappa_u_query,
        x_obs=analysis_df.loc[has_u, "age"],
        y_obs=analysis_df.loc[has_u, "understood"],
        n_trials=context.model_data.n_trials,
        hdi_prob=context.reporting.hdi,
        output_dir=context.reporting.output_dir,
        suffix="u",
        outcome_label="Words understood",
        y_label="Predicted words understood",
    )

    # ---- Per-outcome plots: spoken ----

    _run_bivariate_outcome_plots(
        samples=samples,
        y_plot=samples.y_s_plot,
        y_query=samples.y_s_query,
        f_plot=samples.f_s_plot,
        kappa_plot=samples.kappa_s_plot,
        kappa_query=samples.kappa_s_query,
        x_obs=analysis_df.loc[has_s, "age"],
        y_obs=analysis_df.loc[has_s, "spoken"],
        n_trials=context.model_data.n_trials,
        hdi_prob=context.reporting.hdi,
        output_dir=context.reporting.output_dir,
        suffix="s",
        outcome_label="Words spoken",
        y_label="Predicted words spoken",
    )


# ============================================================
# Fit orchestration
# ============================================================


def fit_bivariate_model(
    config: str,
    definition: BivariateModelDefinition,
) -> BivariateContext:
    """
    Shared fit pipeline for bivariate models (VG05-VG06).
    """
    run_banner(definition.banner, subtitle=f"sampling config: {config}")

    env_info.report_environment_info()

    console.print()
    package_metadata.report_package_versions(PACKAGE_LIST)

    context: BivariateContext = ModelFitContext(
        reporting=reporting.ReportingConfiguration(
            model_name=definition.model_id,
            config_name=definition.config_name,
            output_root_dir=local_env.OUTPUT_DIR,
            hdi=0.90,
        ),
        sampling=sampling.get_sampling_configuration(config),
    )

    if os.path.exists(context.reporting.output_dir):
        shutil.rmtree(context.reporting.output_dir)

    os.makedirs(context.reporting.output_dir, exist_ok=True)

    timings = context.timings
    run_started = time.perf_counter()

    with section("Prepare data", timings=timings):
        prepare_bivariate_data(context, definition)

    with section("Priors and hyperparameters", timings=timings):
        configure_bivariate_priors(context, definition)

    with section("Model definition and initialisation", timings=timings):
        build_model(context)

    with section("Prior predictive checks", timings=timings):
        prior_predictive_checks(context)

    with section("Posterior sampling", timings=timings):
        sample(context)

    with section("Diagnostics", timings=timings):
        diagnostics(context)

    with section("Posterior predictions", timings=timings):
        sample_posterior_predictive(context, definition)

    with section("Posterior summary", timings=timings):
        posterior_summary(context)

    with section("Plots", timings=timings):
        _run_bivariate_joint_plots(context, definition)

    with section("Report", timings=timings):
        report(context)

    pipeline_summary(
        f"Pipeline summary — {context.reporting.model_label}", timings
    )
    console.print(
        f"[dim]Total wall time: "
        f"{vg_reporting.format_duration(time.perf_counter() - run_started)}[/dim]"
    )

    return context
