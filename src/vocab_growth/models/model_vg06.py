# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Model VG06: Joint model of words understood and spoken (A → U, A → S, U → S)
- typically developing children

Uses a production-ratio reparameterization:
    p_U(a) = sigmoid(f_U(a))
    q(a)   = sigmoid(h(a))       # fraction of understood words spoken
    p_S(a) = p_U(a) * q(a)       # enforces p_S <= p_U by construction
"""

import os
import shutil
from dataclasses import dataclass

import arviz as az
import dse_research_utils.environment.info as env_info
import dse_research_utils.math.constants as math_constants
import dse_research_utils.metadata.packages as package_metadata
import dse_research_utils.plot.diagnostics_mcmc as plot_diagnostics_mcmc
import dse_research_utils.plot.distributions as plot_dist
import dse_research_utils.plot.styles as plot_styles
import dse_research_utils.statistics.descriptive as descriptive_stats
import dse_research_utils.statistics.models.data as model_data
import dse_research_utils.statistics.models.pymc_utils as pymc_utils
import dse_research_utils.statistics.models.reporting as reporting
import dse_research_utils.statistics.models.sampling as sampling
import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import preliz as pz
import pymc as pm
from arviz import InferenceData
from preliz.distributions.distributions import Continuous
from rich import print
from rich.pretty import pprint

import vocab_growth.environment as local_env
import vocab_growth.plotting as plotting
import vocab_growth.posterior_analysis as posterior_analysis
from vocab_growth.models.common import BaseModelConfiguration, ModelFitContext

EPSILON = math_constants.EPSILON


# ============================================================
# VG06-specific dataclasses
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


Vg06Context = ModelFitContext[BivariateModelConfiguration, BivariateModelSamples]


# ============================================================
# Data preparation
# ============================================================


def prepare_model_data(
    n_trials: int = 690,
    sample_fraction: float = 0.1,
    random_seed: int = 47,
) -> tuple[model_data.BinomialModelData, pd.DataFrame, pd.DataFrame]:
    """Load TD data, keeping rows where at least one of understood/spoken is observed."""

    con = duckdb.connect(f"{local_env.DATA_DIR}/vocabulary.duckdb")
    td_df = (
        con.execute(
            """
        SELECT
            form,
            age,
            comprehension                      as understood,
            production                         as spoken,
            typically_developing,
            health_conditions
        FROM wordbank_child
        WHERE typically_developing = true
            AND age < 31
            AND health_conditions IS NULL
            AND (form = 'Oxford CDI' OR form = 'WG' OR form = 'WS')
        """
        )
        .df()
        .sample(frac=sample_fraction, random_state=random_seed)
        .reset_index(drop=True)
    )
    con.close()

    analysis_df = td_df[["age", "understood", "spoken"]].copy()

    # Keep rows where at least one outcome is observed (and age is present)
    analysis_df = analysis_df.dropna(subset=["age"])
    has_u = analysis_df["understood"].notna()
    has_s = analysis_df["spoken"].notna()
    analysis_df = analysis_df[has_u | has_s].reset_index(drop=True)

    desc = descriptive_stats.describe_all(analysis_df, alpha=0.05)

    print(
        "\n[green]------------------------------------------------------------[/green]"
    )
    print("[bold green]Descriptive statistics[/bold green]")
    print("[green]------------------------------------------------------------[/green]")

    pprint(desc)

    n = len(analysis_df)
    n_u = int(analysis_df["understood"].notna().sum())
    n_s = int(analysis_df["spoken"].notna().sum())
    n_both = int(
        (analysis_df["understood"].notna() & analysis_df["spoken"].notna()).sum()
    )

    print(f"\n  Total observations:       {n}")
    print(f"  Understood observed:      {n_u}")
    print(f"  Spoken observed:          {n_s}")
    print(f"  Both observed:            {n_both}")
    print(f"  Understood only:          {n_u - n_both}")
    print(f"  Spoken only:              {n_s - n_both}")

    # Create a BinomialModelData for the context interface (using understood as primary)
    X_obs = np.asarray(analysis_df["age"], dtype=float).reshape(-1, 1)
    y_u_valid = analysis_df.loc[analysis_df["understood"].notna(), "understood"]
    y_obs_placeholder = np.zeros(n, dtype=int)
    y_obs_placeholder[analysis_df["understood"].notna().values] = (
        y_u_valid.values.astype(int)
    )

    bmd = model_data.BinomialModelData(
        X_obs=X_obs, y_obs=y_obs_placeholder, n_trials=n_trials
    )

    return bmd, analysis_df, desc


# ============================================================
# Model building
# ============================================================


def build_model(context: Vg06Context):
    """Build the bivariate PyMC model."""
    config = context.model_config

    print(
        "\n[green]------------------------------------------------------------[/green]"
    )
    print("[bold green]Model definition and initialisation[/bold green]")
    print("[green]------------------------------------------------------------[/green]")
    print()

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

    print(f"  Total observations:   {n}")
    print(f"  Understood observed:  {n_u}")
    print(f"  Spoken observed:      {n_s}")
    print(f"  n_trials:             {n_trials}")

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

    print(f"  Age (months) - mean: {X_obs_mean:.2f}, std: {X_obs_std:.2f}")

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

    print(f"  HSGP basis size (m): {M}")
    print(f"  HSGP L: {L}")

    # Slope anchors
    slope_age_a = float(config.slope_anchors[0])
    slope_age_b = float(config.slope_anchors[1])
    slope_age_a_z = (slope_age_a - X_obs_mean) / X_obs_std
    slope_age_b_z = (slope_age_b - X_obs_mean) / X_obs_std

    print(f"  Slope anchors (z-scores): {slope_age_a_z:.2f}, {slope_age_b_z:.2f}")

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
# HSGP hyperparameters
# ============================================================


def get_hsgp_hyperparams(X_obs_z, ell_range_z):
    """Compute HSGP basis size and boundary factor."""
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


def extract_model_samples(trace: InferenceData) -> BivariateModelSamples:
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


def prepare_data(context: Vg06Context):
    """Load and prepare data. Report descriptive statistics."""
    print(
        "\n[green]------------------------------------------------------------[/green]"
    )
    print("[bold green]Prepare data[/bold green]")
    print("[green]------------------------------------------------------------[/green]")
    print()

    data, analysis_df, desc_stats = prepare_model_data(
        n_trials=690,
    )

    context.set_model_data(data, analysis_df)
    context.dataframes["descriptive_stats"] = desc_stats

    desc_stats.to_csv(
        os.path.join(context.reporting.output_dir, "descriptive_statistics.csv"),
        index=True,
    )


def configure_model(context: Vg06Context):
    """Configure priors and hyperparameters."""
    print(
        "\n[green]------------------------------------------------------------[/green]"
    )
    print("[bold green]Priors and hyperparameters[/bold green]")
    print("[green]------------------------------------------------------------[/green]")
    print()

    # --- Understood (U) trajectory priors ---

    ell_unit_u_dist = pz.Beta(alpha=3.0, beta=3.0)
    context.plots["ell_unit_u_dist"] = plot_dist.plot_distribution(
        ell_unit_u_dist, context.reporting.output_dir, "ell_unit_u_dist"
    )
    print(
        f"[bold yellow]ell_unit_u_dist:[/bold yellow] {ell_unit_u_dist.summary(mass=context.reporting.hdi)}"
    )

    eta_u_dist = pz.HalfNormal(sigma=0.4)
    context.plots["eta_u_dist"] = plot_dist.plot_distribution(
        eta_u_dist, context.reporting.output_dir, "eta_u_dist"
    )
    print(
        f"[bold yellow]eta_u_dist:[/bold yellow] {eta_u_dist.summary(mass=context.reporting.hdi)}"
    )

    # Slope priors for understood (from VG04)
    p_slope_low_u_dist = pz.Beta(alpha=1.0, beta=20)
    context.plots["p_slope_low_u_dist"] = plot_dist.plot_distribution(
        p_slope_low_u_dist, context.reporting.output_dir, "p_slope_low_u_dist"
    )
    print(
        f"[bold yellow]p_slope_low_u_dist:[/bold yellow] {p_slope_low_u_dist.summary(mass=context.reporting.hdi)}"
    )

    p_slope_hi_u_dist = pz.Beta(alpha=1.5, beta=1.1)
    context.plots["p_slope_hi_u_dist"] = plot_dist.plot_distribution(
        p_slope_hi_u_dist, context.reporting.output_dir, "p_slope_hi_u_dist"
    )
    print(
        f"[bold yellow]p_slope_hi_u_dist:[/bold yellow] {p_slope_hi_u_dist.summary(mass=context.reporting.hdi)}"
    )

    # --- Production ratio (q) priors ---

    ell_unit_q_dist = pz.Beta(alpha=3.0, beta=3.0)
    context.plots["ell_unit_q_dist"] = plot_dist.plot_distribution(
        ell_unit_q_dist, context.reporting.output_dir, "ell_unit_q_dist"
    )
    print(
        f"[bold yellow]ell_unit_q_dist:[/bold yellow] {ell_unit_q_dist.summary(mass=context.reporting.hdi)}"
    )

    eta_q_dist = pz.HalfNormal(sigma=0.4)
    context.plots["eta_q_dist"] = plot_dist.plot_distribution(
        eta_q_dist, context.reporting.output_dir, "eta_q_dist"
    )
    print(
        f"[bold yellow]eta_q_dist:[/bold yellow] {eta_q_dist.summary(mass=context.reporting.hdi)}"
    )

    p_slope_low_q_dist = pz.Beta(alpha=1.0, beta=1.2)
    context.plots["p_slope_low_q_dist"] = plot_dist.plot_distribution(
        p_slope_low_q_dist, context.reporting.output_dir, "p_slope_low_q_dist"
    )
    print(
        f"[bold yellow]p_slope_low_q_dist:[/bold yellow] {p_slope_low_q_dist.summary(mass=context.reporting.hdi)}"
    )

    p_slope_hi_q_dist = pz.Beta(alpha=1.2, beta=1.0)
    context.plots["p_slope_hi_q_dist"] = plot_dist.plot_distribution(
        p_slope_hi_q_dist, context.reporting.output_dir, "p_slope_hi_q_dist"
    )
    print(
        f"[bold yellow]p_slope_hi_q_dist:[/bold yellow] {p_slope_hi_q_dist.summary(mass=context.reporting.hdi)}"
    )

    # --- Kappa priors — understood ---

    kappa_min_u_dist = pz.LogNormal(mu=np.log(5.0), sigma=0.6)
    context.plots["kappa_min_u_dist"] = plot_dist.plot_distribution(
        kappa_min_u_dist, context.reporting.output_dir, "kappa_min_u_dist"
    )
    print(
        f"[bold yellow]kappa_min_u_dist:[/bold yellow] {kappa_min_u_dist.summary(mass=context.reporting.hdi)}"
    )

    a_kappa_u_dist = pz.Normal(mu=np.log(8.0), sigma=1.0)
    context.plots["a_kappa_u_dist"] = plot_dist.plot_distribution(
        a_kappa_u_dist, context.reporting.output_dir, "a_kappa_u_dist"
    )
    print(
        f"[bold yellow]a_kappa_u_dist:[/bold yellow] {a_kappa_u_dist.summary(mass=context.reporting.hdi)}"
    )

    b_kappa_mag_u_dist = pz.HalfNormal(sigma=0.3)
    context.plots["b_kappa_mag_u_dist"] = plot_dist.plot_distribution(
        b_kappa_mag_u_dist, context.reporting.output_dir, "b_kappa_mag_u_dist"
    )
    print(
        f"[bold yellow]b_kappa_mag_u_dist:[/bold yellow] {b_kappa_mag_u_dist.summary(mass=context.reporting.hdi)}"
    )

    # --- Kappa priors — spoken ---

    kappa_min_s_dist = pz.LogNormal(mu=np.log(5.0), sigma=0.6)
    context.plots["kappa_min_s_dist"] = plot_dist.plot_distribution(
        kappa_min_s_dist, context.reporting.output_dir, "kappa_min_s_dist"
    )
    print(
        f"[bold yellow]kappa_min_s_dist:[/bold yellow] {kappa_min_s_dist.summary(mass=context.reporting.hdi)}"
    )

    a_kappa_s_dist = pz.Normal(mu=np.log(8.0), sigma=1.0)
    context.plots["a_kappa_s_dist"] = plot_dist.plot_distribution(
        a_kappa_s_dist, context.reporting.output_dir, "a_kappa_s_dist"
    )
    print(
        f"[bold yellow]a_kappa_s_dist:[/bold yellow] {a_kappa_s_dist.summary(mass=context.reporting.hdi)}"
    )

    b_kappa_mag_s_dist = pz.HalfNormal(sigma=0.3)
    context.plots["b_kappa_mag_s_dist"] = plot_dist.plot_distribution(
        b_kappa_mag_s_dist, context.reporting.output_dir, "b_kappa_mag_s_dist"
    )
    print(
        f"[bold yellow]b_kappa_mag_s_dist:[/bold yellow] {b_kappa_mag_s_dist.summary(mass=context.reporting.hdi)}"
    )

    # --- Configuration object ---

    slope_age_low = 12
    slope_age_hi = 26

    config = BivariateModelConfiguration(
        slope_anchors=(slope_age_low, slope_age_hi),
        ell_months_range=(2, 12),
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
        n_plot=500,
        ages_query=[9, 12, 15, 18, 21, 24, 27, 30],
    )

    context.set_model_config(config)


def prior_predictive_checks(context: Vg06Context):
    """Run prior predictive checks."""
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


def sample(context: Vg06Context):
    """Draw samples from the posterior using MCMC."""
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
    print("[bold green]Posterior sampling completed.[/bold green]")


def diagnostics(context: Vg06Context):
    """Run diagnostics on the posterior samples."""
    print(
        "\n[green]------------------------------------------------------------[/green]"
    )
    print("[bold green]Diagnostics[/bold green]")
    print("[green]------------------------------------------------------------[/green]")
    print()

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

    # KDE pair plot
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
    print("LOO-CV (words spoken):")
    print(loocv_s)
    print("\nLOO-CV (words understood):")
    print(loocv_u)


def sample_posterior_predictive(context: Vg06Context):
    """Sample from the posterior predictive distribution."""
    print(
        "\n[green]------------------------------------------------------------[/green]"
    )
    print("[bold green]Posterior predictions[/bold green]")
    print("[green]------------------------------------------------------------[/green]")
    print()

    n_trials = context.model_data.n_trials

    p_u_plot = context.model_variables["p_u_plot"]
    p_u_query = context.model_variables["p_u_query"]
    kappa_u_plot = context.model_variables["kappa_u_plot"]
    kappa_u_query = context.model_variables["kappa_u_query"]

    p_s_plot = context.model_variables["p_s_plot"]
    p_s_query = context.model_variables["p_s_query"]
    kappa_s_plot = context.model_variables["kappa_s_plot"]
    kappa_s_query = context.model_variables["kappa_s_query"]

    with context.model:
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


def posterior_summary(context: Vg06Context):
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
    print("\n[bold green]Posterior summary — words understood[/bold green]")
    pprint(summary_u)
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
    print("\n[bold green]Posterior summary — words spoken[/bold green]")
    pprint(summary_s)
    context.dataframes["posterior_summary_s"] = summary_s
    summary_s.to_csv(
        os.path.join(context.reporting.output_dir, "posterior_summary_s.csv"),
        index=False,
    )

    # Production rate summary
    q_query_median = np.median(samples.q_query, axis=1)
    q_query_hdi = az.hdi(samples.q_query.T, hdi_prob=hdi_prob)

    summary_q = pd.DataFrame(
        {
            "age_months": samples.X_query,
            "q_median": q_query_median,
            "q_hdi_lo": q_query_hdi[:, 0],
            "q_hdi_hi": q_query_hdi[:, 1],
        }
    )
    print("\n[bold green]Posterior summary — production rate q(a)[/bold green]")
    pprint(summary_q)
    context.dataframes["posterior_summary_q"] = summary_q
    summary_q.to_csv(
        os.path.join(context.reporting.output_dir, "posterior_summary_q.csv"),
        index=False,
    )


# ============================================================
# VG06-specific plotting functions
# ============================================================


def plot_joint_trajectory(
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
    q_hdi = az.hdi(q_plot.T, hdi_prob=hdi_prob)
    q_hdi_75 = az.hdi(q_plot.T, hdi_prob=0.75)
    q_hdi_50 = az.hdi(q_plot.T, hdi_prob=0.50)

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
    gap_hdi = az.hdi(gap.T, hdi_prob=hdi_prob)
    gap_hdi_50 = az.hdi(gap.T, hdi_prob=0.50)

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

    return fig


def plot_understood_vs_spoken(
    samples: BivariateModelSamples,
    n_trials: int,
    hdi_prob: float = 0.90,
    output_dir: str | None = None,
    filename: str | None = None,
):
    """Plot posterior expected words understood (x) vs words spoken (y)."""
    E_u = samples.p_u_plot * n_trials  # (n_plot, n_samples)
    E_s = samples.p_s_plot * n_trials

    E_u_median = np.median(E_u, axis=1)
    E_s_median = np.median(E_s, axis=1)

    E_s_hdi = az.hdi(E_s.T, hdi_prob=hdi_prob)
    E_s_hdi_50 = az.hdi(E_s.T, hdi_prob=0.50)

    fig, ax = plt.subplots(figsize=plot_styles.FIGSIZE_XL)

    ax.fill_between(
        E_u_median,
        E_s_hdi[:, 0],
        E_s_hdi[:, 1],
        alpha=0.20,
        label=f"{int(hdi_prob * 100)}% HDI",
    )
    ax.fill_between(
        E_u_median,
        E_s_hdi_50[:, 0],
        E_s_hdi_50[:, 1],
        alpha=0.30,
        label="50% HDI",
    )
    ax.plot(E_u_median, E_s_median, lw=3, label="Median")

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

    return fig


# ============================================================
# Report
# ============================================================


def report(context: Vg06Context):
    """Copy output artefacts to the report directory."""

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


# ============================================================
# Fit orchestration
# ============================================================


def fit(config: str) -> Vg06Context:
    print(
        "\n[green]============================================================[/green]"
    )
    print(
        "[bold green]Fitting Model VG06: Joint model of words understood and spoken"
        " (A -> U, A -> S, U -> S) - typically developing[/bold green]"
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

    context: Vg06Context = ModelFitContext(
        reporting=reporting.ReportingConfiguration(
            model_name="VG06",
            config_name="age-understood-spoken-td",
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

    samples = context.model_samples
    analysis_df = context.analysis_df
    has_u = analysis_df["understood"].notna()
    has_s = analysis_df["spoken"].notna()

    # ---- Joint trajectory plot ----

    fig = plot_joint_trajectory(
        samples,
        n_trials=context.model_data.n_trials,
        output_dir=context.reporting.output_dir,
        filename="joint_trajectory",
    )
    context.plots["joint_trajectory"] = fig
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
        hdi_prob=context.reporting.hdi,
        output_dir=context.reporting.output_dir,
        filename="understood_vs_spoken",
    )
    context.plots["understood_vs_spoken"] = fig
    plt.close(fig)

    # ---- Per-outcome plots: understood ----

    plotting.plot_posterior_predictive_count_distributions_by_query_age(
        X_query=samples.X_query,
        y_query=samples.y_u_query,
        n_trials=context.model_data.n_trials,
        output_dir=context.reporting.output_dir,
        filename="posterior_predictive_count_distributions_u",
        x_label="Words understood (count)",
    )

    plotting.plot_posterior_predictive_pmf(
        samples.X_query,
        samples.X_plot,
        samples.y_u_plot,
        context.model_data.n_trials,
        output_dir=context.reporting.output_dir,
        filename="posterior_predictive_pmf_u",
        x_label="Words understood (count)",
    )

    plotting.plot_posterior_predictive_cdf(
        samples.X_query,
        samples.X_plot,
        samples.y_u_plot,
        context.model_data.n_trials,
        output_dir=context.reporting.output_dir,
        filename="posterior_predictive_cdf_u",
        x_label="Words understood (count)",
    )

    plotting.plot_posterior_predictive_median_trend(
        samples.X_plot,
        samples.y_u_plot,
        analysis_df.loc[has_u, "age"],
        analysis_df.loc[has_u, "understood"],
        output_dir=context.reporting.output_dir,
        filename="posterior_predictive_median_trend_u",
        y_label="Predicted words understood",
    )

    plotting.plot_posterior_predictive_median_trend(
        samples.X_plot,
        samples.y_u_plot,
        analysis_df.loc[has_u, "age"],
        analysis_df.loc[has_u, "understood"],
        smooth=True,
        savgol_window_length=15,
        savgol_polyorder=3,
        smooth_intervals=True,
        output_dir=context.reporting.output_dir,
        filename="posterior_predictive_median_trend_u_smoothed",
        y_label="Predicted words understood",
    )

    plotting.plot_expected_learning_rate(
        samples.X_plot,
        samples.f_u_plot,
        n_trials=context.model_data.n_trials,
        hdi_prob=context.reporting.hdi,
        output_dir=context.reporting.output_dir,
        filename="expected_learning_rate_u",
        y_label="Estimated understood word gain per month",
    )

    plotting.plot_expected_learning_rate(
        samples.X_plot,
        samples.f_u_plot,
        n_trials=context.model_data.n_trials,
        hdi_prob=context.reporting.hdi,
        smooth=True,
        savgol_window_length=15,
        savgol_polyorder=3,
        smooth_intervals=True,
        output_dir=context.reporting.output_dir,
        filename="expected_learning_rate_u_smoothed",
        y_label="Estimated understood word gain per month",
    )

    plotting.plot_posterior_kappa(
        samples.X_plot,
        samples.kappa_u_plot,
        samples.X_query,
        samples.kappa_u_query,
        n_trials=context.model_data.n_trials,
        hdi_prob=context.reporting.hdi,
        output_dir=context.reporting.output_dir,
        filename="posterior_kappa_u",
    )

    # ---- Per-outcome plots: spoken ----

    plotting.plot_posterior_predictive_count_distributions_by_query_age(
        X_query=samples.X_query,
        y_query=samples.y_s_query,
        n_trials=context.model_data.n_trials,
        output_dir=context.reporting.output_dir,
        filename="posterior_predictive_count_distributions_s",
    )

    plotting.plot_posterior_predictive_pmf(
        samples.X_query,
        samples.X_plot,
        samples.y_s_plot,
        context.model_data.n_trials,
        output_dir=context.reporting.output_dir,
        filename="posterior_predictive_pmf_s",
        x_label="Words spoken (count)",
    )

    plotting.plot_posterior_predictive_cdf(
        samples.X_query,
        samples.X_plot,
        samples.y_s_plot,
        context.model_data.n_trials,
        output_dir=context.reporting.output_dir,
        filename="posterior_predictive_cdf_s",
        x_label="Words spoken (count)",
    )

    plotting.plot_posterior_predictive_median_trend(
        samples.X_plot,
        samples.y_s_plot,
        analysis_df.loc[has_s, "age"],
        analysis_df.loc[has_s, "spoken"],
        output_dir=context.reporting.output_dir,
        filename="posterior_predictive_median_trend_s",
        y_label="Predicted words spoken",
    )

    plotting.plot_posterior_predictive_median_trend(
        samples.X_plot,
        samples.y_s_plot,
        analysis_df.loc[has_s, "age"],
        analysis_df.loc[has_s, "spoken"],
        smooth=True,
        savgol_window_length=15,
        savgol_polyorder=3,
        smooth_intervals=True,
        output_dir=context.reporting.output_dir,
        filename="posterior_predictive_median_trend_s_smoothed",
        y_label="Predicted words spoken",
    )

    plotting.plot_expected_learning_rate(
        samples.X_plot,
        samples.f_s_plot,
        n_trials=context.model_data.n_trials,
        hdi_prob=context.reporting.hdi,
        output_dir=context.reporting.output_dir,
        filename="expected_learning_rate_s",
        y_label="Estimated spoken word gain per month",
    )

    plotting.plot_expected_learning_rate(
        samples.X_plot,
        samples.f_s_plot,
        n_trials=context.model_data.n_trials,
        hdi_prob=context.reporting.hdi,
        smooth=True,
        savgol_window_length=15,
        savgol_polyorder=3,
        smooth_intervals=True,
        output_dir=context.reporting.output_dir,
        filename="expected_learning_rate_s_smoothed",
        y_label="Estimated spoken word gain per month",
    )

    plotting.plot_posterior_kappa(
        samples.X_plot,
        samples.kappa_s_plot,
        samples.X_query,
        samples.kappa_s_query,
        n_trials=context.model_data.n_trials,
        hdi_prob=context.reporting.hdi,
        output_dir=context.reporting.output_dir,
        filename="posterior_kappa_s",
    )

    report(context)

    return context
