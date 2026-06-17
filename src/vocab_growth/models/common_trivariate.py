# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Shared dataclasses and pipeline functions for the trivariate vocabulary growth
model (VG14): words understood, spoken and signed.

Uses a production-ratio reparameterization that extends the bivariate engine
(common_bivariate.py) with a third parallel ratio for signing:

    p_U(a)    = sigmoid(f_U(a))                        # proportion understood
    q(a)      = sigmoid(h(a))                          # fraction of understood spoken
    r(a)      = sigmoid(g_sign(a))                     # fraction of understood signed
    p_S(a)    = p_U(a) * q(a)                          # enforces p_S    <= p_U
    p_Sign(a) = p_U(a) * r(a)                          # enforces p_Sign <= p_U
    p_any(a)  = p_U(a) * (1 - (1 - r(a)) * (1 - q(a))) # total expressive vocabulary

The total-expressive quantity p_any assumes signing and speaking are
*conditionally independent given age* (the stated Option 1 limitation of
issue #49; VG15 will relax it).

This module is a deliberate, self-contained copy-and-extend of
common_bivariate.py. It does not import from or modify the bivariate engine;
some duplication is intentional, to keep the signed logic fully isolated.
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
from vocab_growth.models.definitions import TrivariateModelDefinition
from vocab_growth.models.diagnostics_utils import capped_plot_var_names
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

# Study identifier (in the merged `vocab_combined` view) for the uk_06 dataset.
# The `include_uk06` flag controls whether uk_06's `signed` counts are included in
# the signed likelihood (useful for sensitivity checks / coding comparability).
UK06_STUDY_ID = "uk_06"

# Age window (months) over which signing production is actually observed in the
# data; the reported signed-ratio / crossover curves shade ages outside this as
# extrapolation (no signing data < ~18 mo; data thins out past ~54 mo).
SIGNED_SUPPORT_MONTHS = (18.0, 54.0)


# ============================================================
# Trivariate-specific dataclasses
# ============================================================


@dataclass
class TrivariateModelConfiguration(BaseModelConfiguration):
    """Configuration for the trivariate (understood + spoken + signed) model."""

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

    # Signed ratio (r) priors — intercept-only mean (no age slope)
    intercept_sign_dist: Continuous
    ell_unit_sign_dist: Continuous
    eta_sign_dist: Continuous

    # Kappa priors — understood
    kappa_min_u_dist: Continuous
    a_kappa_u_dist: Continuous
    b_kappa_mag_u_dist: Continuous

    # Kappa priors — spoken
    kappa_min_s_dist: Continuous
    a_kappa_s_dist: Continuous
    b_kappa_mag_s_dist: Continuous

    # Kappa priors — signed
    kappa_min_sign_dist: Continuous
    a_kappa_sign_dist: Continuous
    b_kappa_mag_sign_dist: Continuous


@dataclass
class TrivariateModelSamples:
    """Posterior and predictive samples from the trivariate model."""

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

    # Signed ratio (r) samples
    g_sign_obs: np.ndarray
    g_sign_plot: np.ndarray
    g_sign_query: np.ndarray
    r_obs: np.ndarray
    r_plot: np.ndarray
    r_query: np.ndarray

    # Signed (Sign) samples (derived)
    f_sign_obs: np.ndarray
    f_sign_plot: np.ndarray
    f_sign_query: np.ndarray
    p_sign_obs: np.ndarray
    p_sign_plot: np.ndarray
    p_sign_query: np.ndarray
    y_sign_obs: np.ndarray
    y_sign_plot: np.ndarray
    y_sign_query: np.ndarray
    kappa_sign_plot: np.ndarray
    kappa_sign_query: np.ndarray

    # Total expressive vocabulary (derived; sign ⟂ speech | age)
    p_any_plot: np.ndarray
    p_any_query: np.ndarray

    # Observation masks (over obs_id)
    obs_u_mask: np.ndarray
    """Boolean array: True where understood is observed, shape (n,)."""
    obs_s_mask: np.ndarray
    """Boolean array: True where spoken is observed, shape (n,)."""
    obs_sign_mask: np.ndarray
    """Boolean array: True where signed is observed, shape (n,)."""


TrivariateContext = ModelFitContext[
    TrivariateModelConfiguration, TrivariateModelSamples
]


# ============================================================
# Data preparation
# ============================================================


def prepare_trivariate_data(
    context: TrivariateContext,
    definition: TrivariateModelDefinition,
):
    """Load and prepare data for the trivariate model from its definition."""
    df = vocab_data_utils.load_data(
        population=definition.population,
        columns=["study", "age", "understood", "spoken", "signed"],
        max_age_months=definition.max_age_months,
    )

    # Optionally drop uk_06 `signed` from the signed likelihood (sensitivity / coding
    # comparability check). When dropped, uk_06's understood/spoken counts are retained.
    n_uk06_dropped = 0
    if not definition.include_uk06:
        uk06_signed = (df["study"] == UK06_STUDY_ID) & df["signed"].notna()
        n_uk06_dropped = int(uk06_signed.sum())
        df.loc[uk06_signed, "signed"] = np.nan

    analysis_df = df[["age", "understood", "spoken", "signed"]].copy()

    # Keep rows where at least one outcome is observed (and age is present)
    analysis_df = analysis_df.dropna(subset=["age"])
    has_u = analysis_df["understood"].notna()
    has_s = analysis_df["spoken"].notna()
    has_sign = analysis_df["signed"].notna()
    analysis_df = analysis_df[has_u | has_s | has_sign].reset_index(drop=True)

    desc = descriptive_stats.describe_all(analysis_df, alpha=0.05)

    n = len(analysis_df)
    n_u = int(analysis_df["understood"].notna().sum())
    n_s = int(analysis_df["spoken"].notna().sum())
    n_sign = int(analysis_df["signed"].notna().sum())

    key_value_table(
        "Observation counts",
        [
            ("Total observations", n),
            ("Understood observed", n_u),
            ("Spoken observed", n_s),
            ("Signed observed", n_sign),
            (f"uk_06 signed dropped (include_uk06={definition.include_uk06})", n_uk06_dropped),
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


def configure_trivariate_priors(
    context: TrivariateContext,
    definition: TrivariateModelDefinition,
):
    """Configure priors and hyperparameters from a trivariate model definition."""
    # --- Understood (U) trajectory priors ---
    heading("Understood trajectory priors", style="bold cyan")

    ell_unit_u_dist = pz.Beta(
        alpha=definition.ell_unit_u_alpha, beta=definition.ell_unit_u_beta
    )
    _plot_and_print_dist(context, ell_unit_u_dist, "ell_unit_u_dist")

    eta_u_dist = pz.HalfNormal(sigma=definition.eta_u_sigma)
    _plot_and_print_dist(context, eta_u_dist, "eta_u_dist")

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

    # --- Signed ratio (r) priors ---
    heading("Signed ratio priors", style="bold cyan")

    ell_unit_sign_dist = pz.Beta(
        alpha=definition.ell_unit_sign_alpha, beta=definition.ell_unit_sign_beta
    )
    _plot_and_print_dist(context, ell_unit_sign_dist, "ell_unit_sign_dist")

    eta_sign_dist = pz.HalfNormal(sigma=definition.eta_sign_sigma)
    _plot_and_print_dist(context, eta_sign_dist, "eta_sign_dist")

    # Intercept-only signed mean (no age slope): a single weakly-informative
    # intercept on the logit scale lets the data set the signed level / tail.
    intercept_sign_dist = pz.Normal(
        mu=definition.intercept_sign_mu, sigma=definition.intercept_sign_sigma
    )
    _plot_and_print_dist(context, intercept_sign_dist, "intercept_sign_dist")

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

    # --- Kappa priors — signed ---
    heading("Kappa priors — signed", style="bold cyan")

    kp_sign = definition.kappa_sign
    kappa_min_sign_dist = pz.LogNormal(
        mu=kp_sign.kappa_min_mu, sigma=kp_sign.kappa_min_sigma
    )
    _plot_and_print_dist(context, kappa_min_sign_dist, "kappa_min_sign_dist")

    a_kappa_sign_dist = pz.Normal(mu=kp_sign.a_kappa_mu, sigma=kp_sign.a_kappa_sigma)
    _plot_and_print_dist(context, a_kappa_sign_dist, "a_kappa_sign_dist")

    b_kappa_mag_sign_dist = pz.HalfNormal(sigma=kp_sign.b_kappa_mag_sigma)
    _plot_and_print_dist(context, b_kappa_mag_sign_dist, "b_kappa_mag_sign_dist")

    # --- Configuration object ---

    config = TrivariateModelConfiguration(
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
        # Signed rate (intercept-only mean)
        intercept_sign_dist=intercept_sign_dist,
        ell_unit_sign_dist=ell_unit_sign_dist,
        eta_sign_dist=eta_sign_dist,
        # Kappa — understood
        kappa_min_u_dist=kappa_min_u_dist,
        a_kappa_u_dist=a_kappa_u_dist,
        b_kappa_mag_u_dist=b_kappa_mag_u_dist,
        # Kappa — spoken
        kappa_min_s_dist=kappa_min_s_dist,
        a_kappa_s_dist=a_kappa_s_dist,
        b_kappa_mag_s_dist=b_kappa_mag_s_dist,
        # Kappa — signed
        kappa_min_sign_dist=kappa_min_sign_dist,
        a_kappa_sign_dist=a_kappa_sign_dist,
        b_kappa_mag_sign_dist=b_kappa_mag_sign_dist,
        n_plot=definition.n_plot,
        ages_query=definition.ages_query,
    )

    context.set_model_config(config)


# ============================================================
# Model building
# ============================================================


def build_model(context: TrivariateContext):
    """Build the trivariate PyMC model."""
    config = context.model_config

    analysis_df = context.analysis_df

    # Observation masks
    has_u = analysis_df["understood"].notna().values
    has_s = analysis_df["spoken"].notna().values
    has_sign = analysis_df["signed"].notna().values

    X_obs = np.asarray(analysis_df["age"], dtype=float).reshape(-1, 1)
    y_u_observed = np.asarray(analysis_df.loc[has_u, "understood"], dtype=int)
    y_s_observed = np.asarray(analysis_df.loc[has_s, "spoken"], dtype=int)
    y_sign_observed = np.asarray(analysis_df.loc[has_sign, "signed"], dtype=int)

    idx_u = np.where(has_u)[0]
    idx_s = np.where(has_s)[0]
    idx_sign = np.where(has_sign)[0]

    n = len(X_obs)
    n_u = len(y_u_observed)
    n_s = len(y_s_observed)
    n_sign = len(y_sign_observed)
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
    if not np.all(y_sign_observed >= 0):
        raise ValueError("y_sign contains negative counts.")
    if not np.all(y_sign_observed <= n_trials):
        raise ValueError("y_sign exceeds n_trials.")

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
            ("Signed observed", n_sign),
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
        "obs_sign_id": np.arange(n_sign),
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

        # Store masks for extraction
        _ = pm.Data("obs_u_mask", has_u.astype(int), dims=("obs_id",))
        _ = pm.Data("obs_s_mask", has_s.astype(int), dims=("obs_id",))
        _ = pm.Data("obs_sign_mask", has_sign.astype(int), dims=("obs_id",))

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
        # Full-grid (n_all,) intermediates are kept as plain tensors rather than
        # stored Deterministics: only the obs/plot/query slices below are
        # extracted, so storing the full-grid arrays for every draw would waste
        # a large amount of trace memory.
        g_u = eta_u * g_unit_u

        f_u_all = mean_trend_u + g_u

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
        g_q = eta_q * g_unit_q

        h_all = mean_trend_q + g_q

        # ============================================================
        # Signed ratio: g_sign(a) -> r(a) = sigmoid(g_sign(a))
        # ============================================================

        # Intercept-only mean (no age slope): structurally prevents a free slope
        # from extrapolating the signed ratio below the data floor (< ~18 mo),
        # while a wide intercept prior lets the data set the level. The GP carries
        # the age-varying (rise-then-fall) shape.
        intercept_sign = config.intercept_sign_dist.to_pymc("intercept_sign")
        mean_trend_sign = intercept_sign

        # GP for signed rate
        ell_unit_sign = config.ell_unit_sign_dist.to_pymc("ell_unit_sign")
        ell_sign = pm.Deterministic(
            "ell_sign", ell_low_z + (ell_high_z - ell_low_z) * ell_unit_sign
        )
        eta_sign = config.eta_sign_dist.to_pymc("eta_sign")

        cov_sign = pm.gp.cov.ExpQuad(1, ls=ell_sign)
        hsgp_sign = pm.gp.HSGP(cov_func=cov_sign, m=M, L=L)
        g_unit_sign = hsgp_sign.prior("g_unit_sign", X=X_all_z_data, dims="all_id")
        gp_sign = eta_sign * g_unit_sign

        g_sign_all = mean_trend_sign + gp_sign

        # ============================================================
        # Derived quantities: p_U, q, p_S, r, p_Sign, p_any
        # (full-grid quantities are plain tensors; only slices are stored)
        # ============================================================

        p_u_all = pm.math.sigmoid(f_u_all)
        q_all = pm.math.sigmoid(h_all)
        r_all = pm.math.sigmoid(g_sign_all)

        p_s_all = p_u_all * q_all
        p_sign_all = p_u_all * r_all

        # Total expressive vocabulary (sign ⟂ speech | age)
        p_any_all = p_u_all * (1 - (1 - r_all) * (1 - q_all))

        # f_S, f_Sign derived for diagnostics/plotting
        p_s_all_clip = pm.math.clip(p_s_all, EPSILON, 1 - EPSILON)
        f_s_all = pm.math.log(p_s_all_clip) - pm.math.log(1 - p_s_all_clip)
        p_sign_all_clip = pm.math.clip(p_sign_all, EPSILON, 1 - EPSILON)
        f_sign_all = pm.math.log(p_sign_all_clip) - pm.math.log(1 - p_sign_all_clip)

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

        # Signed rate
        _ = pm.Deterministic(
            "g_sign_obs", g_sign_all[i_obs0:i_obs1], dims=("obs_id",)
        )
        _ = pm.Deterministic(
            "g_sign_plot", g_sign_all[i_plot0:i_plot1], dims=("plot_id",)
        )
        _ = pm.Deterministic(
            "g_sign_query", g_sign_all[i_query0:i_query1], dims=("query_id",)
        )

        _ = pm.Deterministic("r_obs", r_all[i_obs0:i_obs1], dims=("obs_id",))
        _ = pm.Deterministic("r_plot", r_all[i_plot0:i_plot1], dims=("plot_id",))
        _ = pm.Deterministic("r_query", r_all[i_query0:i_query1], dims=("query_id",))

        # Signed (derived)
        p_sign_obs = pm.Deterministic(
            "p_sign_obs", p_sign_all[i_obs0:i_obs1], dims=("obs_id",)
        )
        _ = pm.Deterministic(
            "p_sign_plot", p_sign_all[i_plot0:i_plot1], dims=("plot_id",)
        )
        _ = pm.Deterministic(
            "p_sign_query", p_sign_all[i_query0:i_query1], dims=("query_id",)
        )

        _ = pm.Deterministic(
            "f_sign_obs", f_sign_all[i_obs0:i_obs1], dims=("obs_id",)
        )
        _ = pm.Deterministic(
            "f_sign_plot", f_sign_all[i_plot0:i_plot1], dims=("plot_id",)
        )
        _ = pm.Deterministic(
            "f_sign_query", f_sign_all[i_query0:i_query1], dims=("query_id",)
        )

        # Total expressive vocabulary (derived)
        _ = pm.Deterministic(
            "p_any_plot", p_any_all[i_plot0:i_plot1], dims=("plot_id",)
        )
        _ = pm.Deterministic(
            "p_any_query", p_any_all[i_query0:i_query1], dims=("query_id",)
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
        # Kappa — signed
        # ============================================================

        kappa_min_sign = config.kappa_min_sign_dist.to_pymc("kappa_min_sign")
        a_kappa_sign = config.a_kappa_sign_dist.to_pymc("a_kappa_sign")
        b_kappa_mag_sign = config.b_kappa_mag_sign_dist.to_pymc("b_kappa_mag_sign")
        b_kappa_sign = pm.Deterministic("b_kappa_sign", -b_kappa_mag_sign)

        def kappa_sign_of_z(z):
            return kappa_min_sign + pm.math.exp(a_kappa_sign + b_kappa_sign * z)

        kappa_sign_obs = pm.Deterministic(
            "kappa_sign_obs", kappa_sign_of_z(z_obs), dims="obs_id"
        )
        _ = pm.Deterministic(
            "kappa_sign_plot", kappa_sign_of_z(z_plot), dims="plot_id"
        )
        _ = pm.Deterministic(
            "kappa_sign_query", kappa_sign_of_z(z_query), dims="query_id"
        )

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

        # Signed likelihood (only where observed)
        p_sign_obs_sel = p_sign_obs[idx_sign]
        p_sign_obs_clip = pm.math.clip(p_sign_obs_sel, EPSILON, 1 - EPSILON)
        alpha_sign = p_sign_obs_clip * kappa_sign_obs[idx_sign]
        beta_sign = (1 - p_sign_obs_clip) * kappa_sign_obs[idx_sign]

        _ = pm.BetaBinomial(
            "y_sign_obs",
            n=n_trials,
            alpha=alpha_sign,
            beta=beta_sign,
            observed=y_sign_observed,
            dims=("obs_sign_id",),
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


def extract_model_samples(trace: xr.DataTree) -> TrivariateModelSamples:
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

    # Signed rate
    g_sign_obs = _extract_posterior(trace, "g_sign_obs", "obs_id")
    g_sign_plot = _extract_posterior(trace, "g_sign_plot", "plot_id")
    g_sign_query = _extract_posterior(trace, "g_sign_query", "query_id")

    r_obs = _extract_posterior(trace, "r_obs", "obs_id")
    r_plot = _extract_posterior(trace, "r_plot", "plot_id")
    r_query = _extract_posterior(trace, "r_query", "query_id")

    # Signed (derived)
    f_sign_obs = _extract_posterior(trace, "f_sign_obs", "obs_id")
    f_sign_plot = _extract_posterior(trace, "f_sign_plot", "plot_id")
    f_sign_query = _extract_posterior(trace, "f_sign_query", "query_id")

    p_sign_obs = _extract_posterior(trace, "p_sign_obs", "obs_id")
    p_sign_plot = _extract_posterior(trace, "p_sign_plot", "plot_id")
    p_sign_query = _extract_posterior(trace, "p_sign_query", "query_id")

    kappa_sign_plot = _extract_posterior(trace, "kappa_sign_plot", "plot_id")
    kappa_sign_query = _extract_posterior(trace, "kappa_sign_query", "query_id")

    # Total expressive vocabulary (derived)
    p_any_plot = _extract_posterior(trace, "p_any_plot", "plot_id")
    p_any_query = _extract_posterior(trace, "p_any_query", "query_id")

    # Observed data — expand to full obs_id length with NaN where unobserved
    obs_u_mask = np.array(trace.constant_data["obs_u_mask"].values, dtype=bool)
    obs_s_mask = np.array(trace.constant_data["obs_s_mask"].values, dtype=bool)
    obs_sign_mask = np.array(trace.constant_data["obs_sign_mask"].values, dtype=bool)
    n_obs = len(obs_u_mask)

    y_u_obs_raw = np.array(trace.observed_data["y_u_obs"].values, dtype=float)
    y_u_obs = np.full(n_obs, np.nan)
    y_u_obs[obs_u_mask] = y_u_obs_raw

    y_s_obs_raw = np.array(trace.observed_data["y_s_obs"].values, dtype=float)
    y_s_obs = np.full(n_obs, np.nan)
    y_s_obs[obs_s_mask] = y_s_obs_raw

    y_sign_obs_raw = np.array(trace.observed_data["y_sign_obs"].values, dtype=float)
    y_sign_obs = np.full(n_obs, np.nan)
    y_sign_obs[obs_sign_mask] = y_sign_obs_raw

    # Posterior predictive
    y_u_plot = _extract_posterior_predictive(trace, "y_u_plot", "plot_id")
    y_u_query = _extract_posterior_predictive(trace, "y_u_query", "query_id")
    y_s_plot = _extract_posterior_predictive(trace, "y_s_plot", "plot_id")
    y_s_query = _extract_posterior_predictive(trace, "y_s_query", "query_id")
    y_sign_plot = _extract_posterior_predictive(trace, "y_sign_plot", "plot_id")
    y_sign_query = _extract_posterior_predictive(trace, "y_sign_query", "query_id")

    # Constant data
    X_obs = np.array(trace.constant_data["X_obs"].values)
    X_plot = np.array(trace.constant_data["X_plot"].values)
    X_query = np.array(trace.constant_data["X_query"].values)

    # Standardised ages
    X_obs_z = _extract_posterior(trace, "z_obs", "obs_id")
    X_plot_z = _extract_posterior(trace, "z_plot", "plot_id")
    X_query_z = _extract_posterior(trace, "z_query", "query_id")

    return TrivariateModelSamples(
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
        g_sign_obs=g_sign_obs,
        g_sign_plot=g_sign_plot,
        g_sign_query=g_sign_query,
        r_obs=r_obs,
        r_plot=r_plot,
        r_query=r_query,
        f_sign_obs=f_sign_obs,
        f_sign_plot=f_sign_plot,
        f_sign_query=f_sign_query,
        p_sign_obs=p_sign_obs,
        p_sign_plot=p_sign_plot,
        p_sign_query=p_sign_query,
        y_sign_obs=y_sign_obs,
        y_sign_plot=y_sign_plot,
        y_sign_query=y_sign_query,
        kappa_sign_plot=kappa_sign_plot,
        kappa_sign_query=kappa_sign_query,
        p_any_plot=p_any_plot,
        p_any_query=p_any_query,
        obs_u_mask=obs_u_mask,
        obs_s_mask=obs_s_mask,
        obs_sign_mask=obs_sign_mask,
    )


# ============================================================
# Pipeline steps
# ============================================================


def _plot_ratio_prior_samples(context, prior_samples, var_name, y_label, filename):
    """Plot prior-sample curves for a production/signed ratio (q or r)."""
    ratio_samples = (
        prior_samples.prior[var_name]
        .stack(sample=("chain", "draw"))
        .transpose("plot_id", "sample")
    )
    fig = plotting.plot_prior_samples_ratio(
        prior_samples.constant_data["X_plot"].values,
        ratio_samples.values,
        y_label=y_label,
        filename=filename,
        output_dir=context.reporting.output_dir,
    )
    context.plots[filename] = fig
    plt.close(fig)


def prior_predictive_checks(context: TrivariateContext):
    """Run prior predictive checks."""
    with context.model:
        prior_samples = pm.sample_prior_predictive(
            draws=1000,
            random_seed=context.sampling.random_seed,
            compile_kwargs=dict(mode="FAST_COMPILE"),
        )

    context.set_prior_samples(prior_samples)

    analysis_df = context.analysis_df
    has_u = analysis_df["understood"].notna()
    has_s = analysis_df["spoken"].notna()
    has_sign = analysis_df["signed"].notna()

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

    # Prior samples for signed trajectory
    p_sign_plot_samples = (
        prior_samples.prior["p_sign_plot"]
        .stack(sample=("chain", "draw"))
        .transpose("plot_id", "sample")
    )
    plotting.plot_prior_samples(
        prior_samples.constant_data["X_plot"].values,
        p_sign_plot_samples.values,
        analysis_df.loc[has_sign, "age"],
        analysis_df.loc[has_sign, "signed"],
        n_trials=context.model_data.n_trials,
        n_curves=1000,
        x_label="Age (months)",
        y_label="Words signed",
        filename="prior_samples_sign",
        output_dir=context.reporting.output_dir,
    )

    # Prior samples for production rate q(a) and signed rate r(a)
    _plot_ratio_prior_samples(
        context, prior_samples, "q_plot", "q(a) = p_S(a) / p_U(a)", "prior_samples_q"
    )
    _plot_ratio_prior_samples(
        context, prior_samples, "r_plot", "r(a) = p_Sign(a) / p_U(a)", "prior_samples_r"
    )


def sample(context: TrivariateContext):
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


def diagnostics(context: TrivariateContext):
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
    pair_plot_var_names = capped_plot_var_names(
        context.trace,
        var_names,
        squared=True,
    )
    if pair_plot_var_names:
        plot_diagnostics_mcmc.plot_kde_pair(
            context.trace,
            var_names=pair_plot_var_names,
            output_dir=context.reporting.output_dir,
            filename="pair_plot",
        )
        context.plots["pair_plot"] = plt.gcf()
        plt.close()

    # Trace plot
    var_names_ext = var_names + ["kappa_u_obs", "kappa_s_obs", "kappa_sign_obs"]
    trace_var_names = capped_plot_var_names(context.trace, var_names_ext)

    az.plot_trace(
        context.trace,
        var_names=trace_var_names,
        figure_kwargs={"figsize": plot_styles.FIGSIZE_XL},
    )
    plt.savefig(os.path.join(context.reporting.output_dir, "trace_plot.png"), dpi=300)
    context.plots["trace_plot"] = plt.gcf()
    plt.close()

    # Energy plot
    az.plot_energy(
        context.trace,
        figure_kwargs={"figsize": plot_styles.FIGSIZE_XL},
    )
    plt.savefig(os.path.join(context.reporting.output_dir, "energy_plot.png"), dpi=300)
    context.plots["energy_plot"] = plt.gcf()
    plt.close()

    # Posterior densities
    posterior_var_names = capped_plot_var_names(context.trace, var_names)
    az.plot_dist(
        context.trace,
        var_names=posterior_var_names,
        point_estimate="median",
        ci_kind="hdi",
        ci_prob=context.reporting.hdi,
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

    loocv_u = az.loo(context.trace, var_name="y_u_obs")
    loocv_s = az.loo(context.trace, var_name="y_s_obs")
    loocv_sign = az.loo(context.trace, var_name="y_sign_obs")
    context.set_loocv(
        {"y_u_obs": loocv_u, "y_s_obs": loocv_s, "y_sign_obs": loocv_sign}
    )
    heading("LOO-CV — words understood", style="bold cyan")
    console.print(loocv_u)
    heading("LOO-CV — words spoken", style="bold cyan")
    console.print(loocv_s)
    heading("LOO-CV — words signed", style="bold cyan")
    console.print(loocv_sign)


def sample_posterior_predictive(context: TrivariateContext, definition=None):
    """Sample from the posterior predictive distribution.

    VG14 has no random intercepts (it mirrors VG05), so the plot/query
    predictive nodes use the population-mean conditional probabilities directly.
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

    p_sign_plot = context.model_variables["p_sign_plot"]
    p_sign_query = context.model_variables["p_sign_query"]
    kappa_sign_plot = context.model_variables["kappa_sign_plot"]
    kappa_sign_query = context.model_variables["kappa_sign_query"]

    with context.model:
        # Understood — plot / query
        p_u_plot_clip = pm.math.clip(p_u_plot, EPSILON, 1 - EPSILON)
        pm.BetaBinomial(
            "y_u_plot",
            n=n_trials,
            alpha=p_u_plot_clip * kappa_u_plot,
            beta=(1 - p_u_plot_clip) * kappa_u_plot,
            dims=("plot_id",),
        )
        p_u_query_clip = pm.math.clip(p_u_query, EPSILON, 1 - EPSILON)
        pm.BetaBinomial(
            "y_u_query",
            n=n_trials,
            alpha=p_u_query_clip * kappa_u_query,
            beta=(1 - p_u_query_clip) * kappa_u_query,
            dims=("query_id",),
        )
        # Spoken — plot / query
        p_s_plot_clip = pm.math.clip(p_s_plot, EPSILON, 1 - EPSILON)
        pm.BetaBinomial(
            "y_s_plot",
            n=n_trials,
            alpha=p_s_plot_clip * kappa_s_plot,
            beta=(1 - p_s_plot_clip) * kappa_s_plot,
            dims=("plot_id",),
        )
        p_s_query_clip = pm.math.clip(p_s_query, EPSILON, 1 - EPSILON)
        pm.BetaBinomial(
            "y_s_query",
            n=n_trials,
            alpha=p_s_query_clip * kappa_s_query,
            beta=(1 - p_s_query_clip) * kappa_s_query,
            dims=("query_id",),
        )
        # Signed — plot / query
        p_sign_plot_clip = pm.math.clip(p_sign_plot, EPSILON, 1 - EPSILON)
        pm.BetaBinomial(
            "y_sign_plot",
            n=n_trials,
            alpha=p_sign_plot_clip * kappa_sign_plot,
            beta=(1 - p_sign_plot_clip) * kappa_sign_plot,
            dims=("plot_id",),
        )
        p_sign_query_clip = pm.math.clip(p_sign_query, EPSILON, 1 - EPSILON)
        pm.BetaBinomial(
            "y_sign_query",
            n=n_trials,
            alpha=p_sign_query_clip * kappa_sign_query,
            beta=(1 - p_sign_query_clip) * kappa_sign_query,
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
                "y_sign_plot",
                "y_sign_query",
                "y_sign_obs",
            ],
            extend_inferencedata=True,
            random_seed=context.sampling.random_seed,
        )

    context.set_trace(trace)

    trace.to_netcdf(os.path.join(context.reporting.output_dir, "trace.nc"))

    sample_data = extract_model_samples(context.trace)
    context.set_model_samples(sample_data)


def _ratio_summary(X_query, ratio_query, hdi_prob):
    """Median + HDI summary for a ratio (q or r) at query ages."""
    median = np.median(ratio_query, axis=1)
    hdi = az.hdi(ratio_query, prob=hdi_prob)
    return pd.DataFrame(
        {
            "age_months": X_query,
            "median": median,
            "hdi_lo": hdi[:, 0],
            "hdi_hi": hdi[:, 1],
        }
    )


def posterior_summary(context: TrivariateContext):
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

    # Signed summary
    summary_sign = posterior_analysis.posterior_summary_table(
        samples.X_query,
        samples.p_sign_query,
        samples.y_sign_query,
        n_trials=n_trials,
        hdi_prob=hdi_prob,
    )
    dataframe_table(
        summary_sign, title="Posterior summary — words signed", show_index=False
    )
    context.dataframes["posterior_summary_sign"] = summary_sign
    summary_sign.to_csv(
        os.path.join(context.reporting.output_dir, "posterior_summary_sign.csv"),
        index=False,
    )

    # Production rate q(a) summary
    summary_q = _ratio_summary(samples.X_query, samples.q_query, hdi_prob)
    summary_q = summary_q.rename(
        columns={"median": "q_median", "hdi_lo": "q_hdi_lo", "hdi_hi": "q_hdi_hi"}
    )
    dataframe_table(
        summary_q, title="Posterior summary — production rate q(a)", show_index=False
    )
    context.dataframes["posterior_summary_q"] = summary_q
    summary_q.to_csv(
        os.path.join(context.reporting.output_dir, "posterior_summary_q.csv"),
        index=False,
    )

    # Signed rate r(a) summary
    summary_r = _ratio_summary(samples.X_query, samples.r_query, hdi_prob)
    summary_r = summary_r.rename(
        columns={"median": "r_median", "hdi_lo": "r_hdi_lo", "hdi_hi": "r_hdi_hi"}
    )
    dataframe_table(
        summary_r, title="Posterior summary — signed rate r(a)", show_index=False
    )
    context.dataframes["posterior_summary_r"] = summary_r
    summary_r.to_csv(
        os.path.join(context.reporting.output_dir, "posterior_summary_r.csv"),
        index=False,
    )

    # Total expressive vocabulary p_any(a) summary (expected count; no
    # predictive count node exists for this derived quantity).
    Ey_any = samples.p_any_query * n_trials
    Ey_any_median = np.median(Ey_any, axis=1)
    Ey_any_hdi = az.hdi(Ey_any, prob=hdi_prob)
    p_any_median = np.median(samples.p_any_query, axis=1)
    p_any_hdi = az.hdi(samples.p_any_query, prob=hdi_prob)
    summary_p_any = pd.DataFrame(
        {
            "age_months": samples.X_query,
            "p_any_median": p_any_median,
            "p_any_hdi_lo": p_any_hdi[:, 0],
            "p_any_hdi_hi": p_any_hdi[:, 1],
            "Ey_any_median": Ey_any_median,
            "Ey_any_hdi_lo": Ey_any_hdi[:, 0],
            "Ey_any_hdi_hi": Ey_any_hdi[:, 1],
        }
    )
    dataframe_table(
        summary_p_any,
        title="Posterior summary — total expressive vocabulary p_any(a)",
        show_index=False,
    )
    context.dataframes["posterior_summary_p_any"] = summary_p_any
    summary_p_any.to_csv(
        os.path.join(context.reporting.output_dir, "posterior_summary_p_any.csv"),
        index=False,
    )


# ============================================================
# Trivariate-specific plotting functions
# ============================================================


def plot_understood_spoken_signed_trajectory(
    samples: TrivariateModelSamples,
    n_trials: int,
    output_dir: str | None = None,
    filename: str | None = None,
):
    """Plot understood, spoken and signed posterior predictive median trends."""
    X_plot = samples.X_plot

    y_u_median = np.quantile(samples.y_u_plot, 0.50, axis=1)
    y_u_90 = np.quantile(samples.y_u_plot, [0.05, 0.95], axis=1).T

    y_s_median = np.quantile(samples.y_s_plot, 0.50, axis=1)
    y_s_90 = np.quantile(samples.y_s_plot, [0.05, 0.95], axis=1).T

    y_sign_median = np.quantile(samples.y_sign_plot, 0.50, axis=1)
    y_sign_90 = np.quantile(samples.y_sign_plot, [0.05, 0.95], axis=1).T

    fig, ax = plt.subplots(figsize=plot_styles.FIGSIZE_XL)

    ax.fill_between(X_plot, y_u_90[:, 0], y_u_90[:, 1], alpha=0.15, color="C0")
    ax.plot(X_plot, y_u_median, lw=3, color="C0", label="Words understood (median)")

    ax.fill_between(X_plot, y_s_90[:, 0], y_s_90[:, 1], alpha=0.15, color="C1")
    ax.plot(X_plot, y_s_median, lw=3, color="C1", label="Words spoken (median)")

    ax.fill_between(X_plot, y_sign_90[:, 0], y_sign_90[:, 1], alpha=0.15, color="C2")
    ax.plot(X_plot, y_sign_median, lw=3, color="C2", label="Words signed (median)")

    # Observed data
    X_obs = samples.X_obs
    u_mask = ~np.isnan(samples.y_u_obs)
    if u_mask.any():
        ax.scatter(X_obs[u_mask], samples.y_u_obs[u_mask], s=10, alpha=0.2, color="C0")
    s_mask = ~np.isnan(samples.y_s_obs)
    if s_mask.any():
        ax.scatter(X_obs[s_mask], samples.y_s_obs[s_mask], s=10, alpha=0.2, color="C1")
    sign_mask = ~np.isnan(samples.y_sign_obs)
    if sign_mask.any():
        ax.scatter(
            X_obs[sign_mask], samples.y_sign_obs[sign_mask], s=10, alpha=0.3, color="C2"
        )

    ax.set_xlabel("Age (months)")
    ax.set_ylabel("Word count")
    ax.legend(loc="upper left", frameon=True)
    ax.set_ylim(-20, n_trials + 50)
    ax.set_title("Words understood, spoken and signed")

    if output_dir is not None and filename is not None:
        fig.savefig(os.path.join(output_dir, f"{filename}.png"), dpi=300)
        fig.savefig(os.path.join(output_dir, f"{filename}.svg"))
        _save_csv(
            pd.DataFrame(
                {
                    "age_months": X_plot,
                    "understood_median": y_u_median,
                    "understood_p05": y_u_90[:, 0],
                    "understood_p95": y_u_90[:, 1],
                    "spoken_median": y_s_median,
                    "spoken_p05": y_s_90[:, 0],
                    "spoken_p95": y_s_90[:, 1],
                    "signed_median": y_sign_median,
                    "signed_p05": y_sign_90[:, 0],
                    "signed_p95": y_sign_90[:, 1],
                }
            ),
            output_dir,
            filename,
        )

    return fig


def _shade_extrapolation(ax, support_range):
    """Shade ages outside the signing-data support window as extrapolation."""
    if support_range is None:
        return
    lo, hi = support_range
    x0, x1 = ax.get_xlim()
    label = "extrapolation (no/sparse signing data)"
    if x0 < lo:
        ax.axvspan(x0, lo, color="grey", alpha=0.12, lw=0, label=label)
        label = None
    if x1 > hi:
        ax.axvspan(hi, x1, color="grey", alpha=0.12, lw=0, label=label)
    ax.set_xlim(x0, x1)


def plot_signed_rate(
    samples: TrivariateModelSamples,
    hdi_prob: float = 0.90,
    output_dir: str | None = None,
    filename: str | None = None,
    support_range: tuple[float, float] | None = None,
):
    """Plot the posterior of the signed ratio r(a) = p_Sign(a) / p_U(a) over age."""
    X_plot = samples.X_plot
    r_plot = samples.r_plot

    r_median = np.median(r_plot, axis=1)
    r_hdi = az.hdi(r_plot, prob=hdi_prob)
    r_hdi_75 = az.hdi(r_plot, prob=0.75)
    r_hdi_50 = az.hdi(r_plot, prob=0.50)

    fig, ax = plt.subplots(figsize=plot_styles.FIGSIZE_XL)

    ax.fill_between(
        X_plot, r_hdi[:, 0], r_hdi[:, 1], alpha=0.20, label=f"{int(hdi_prob * 100)}% HDI"
    )
    ax.fill_between(X_plot, r_hdi_75[:, 0], r_hdi_75[:, 1], alpha=0.25, label="75% HDI")
    ax.fill_between(X_plot, r_hdi_50[:, 0], r_hdi_50[:, 1], alpha=0.30, label="50% HDI")
    ax.plot(X_plot, r_median, lw=3, label="Median r(a)")

    ax.set_xlabel("Age (months)")
    ax.set_ylabel("r(a) = p_Sign(a) / p_U(a)")
    ax.set_ylim(0, 1)
    _shade_extrapolation(ax, support_range)
    ax.legend(loc="upper left", frameon=True)
    ax.set_title("Signed ratio r(a)")

    if output_dir is not None and filename is not None:
        fig.savefig(os.path.join(output_dir, f"{filename}.png"), dpi=300)
        fig.savefig(os.path.join(output_dir, f"{filename}.svg"))
        _save_csv(
            pd.DataFrame(
                {
                    "age_months": X_plot,
                    "r_median": r_median,
                    "hdi_lo": r_hdi[:, 0],
                    "hdi_hi": r_hdi[:, 1],
                    "hdi75_lo": r_hdi_75[:, 0],
                    "hdi75_hi": r_hdi_75[:, 1],
                    "hdi50_lo": r_hdi_50[:, 0],
                    "hdi50_hi": r_hdi_50[:, 1],
                }
            ),
            output_dir,
            filename,
        )

    return fig


def plot_sign_speech_crossover(
    samples: TrivariateModelSamples,
    hdi_prob: float = 0.90,
    output_dir: str | None = None,
    filename: str | None = None,
    support_range: tuple[float, float] | None = None,
):
    """Plot signed rate r(a) against spoken rate q(a) — the sign->speech hand-off."""
    X_plot = samples.X_plot

    q_median = np.median(samples.q_plot, axis=1)
    q_hdi = az.hdi(samples.q_plot, prob=hdi_prob)
    r_median = np.median(samples.r_plot, axis=1)
    r_hdi = az.hdi(samples.r_plot, prob=hdi_prob)

    fig, ax = plt.subplots(figsize=plot_styles.FIGSIZE_XL)

    ax.fill_between(X_plot, q_hdi[:, 0], q_hdi[:, 1], alpha=0.15, color="C1")
    ax.plot(
        X_plot, q_median, lw=3, color="C1", label="Spoken ratio q(a) (fraction spoken)"
    )

    ax.fill_between(X_plot, r_hdi[:, 0], r_hdi[:, 1], alpha=0.15, color="C2")
    ax.plot(
        X_plot, r_median, lw=3, color="C2", label="Signed ratio r(a) (fraction signed)"
    )

    ax.set_xlabel("Age (months)")
    ax.set_ylabel("Fraction of understood words")
    ax.set_ylim(0, 1)
    _shade_extrapolation(ax, support_range)
    ax.legend(loc="upper left", frameon=True)
    ax.set_title("Sign → speech hand-off: r(a) vs q(a)")

    if output_dir is not None and filename is not None:
        fig.savefig(os.path.join(output_dir, f"{filename}.png"), dpi=300)
        fig.savefig(os.path.join(output_dir, f"{filename}.svg"))
        _save_csv(
            pd.DataFrame(
                {
                    "age_months": X_plot,
                    "q_median": q_median,
                    "q_hdi_lo": q_hdi[:, 0],
                    "q_hdi_hi": q_hdi[:, 1],
                    "r_median": r_median,
                    "r_hdi_lo": r_hdi[:, 0],
                    "r_hdi_hi": r_hdi[:, 1],
                }
            ),
            output_dir,
            filename,
        )

    return fig


def plot_modality_trajectories(
    samples: TrivariateModelSamples,
    n_trials: int,
    hdi_prob: float = 0.90,
    output_dir: str | None = None,
    filename: str | None = None,
):
    """Plot expected p_U, p_S, p_Sign and p_any trajectories (in word counts)."""
    X_plot = samples.X_plot

    E_u = np.median(samples.p_u_plot, axis=1) * n_trials
    E_s = np.median(samples.p_s_plot, axis=1) * n_trials
    E_sign = np.median(samples.p_sign_plot, axis=1) * n_trials
    E_any = np.median(samples.p_any_plot, axis=1) * n_trials
    any_hdi = az.hdi(samples.p_any_plot, prob=hdi_prob) * n_trials

    fig, ax = plt.subplots(figsize=plot_styles.FIGSIZE_XL)

    ax.fill_between(X_plot, any_hdi[:, 0], any_hdi[:, 1], alpha=0.12, color="C3")
    ax.plot(X_plot, E_any, lw=3, color="C3", label="Total expressive p_any")
    ax.plot(X_plot, E_u, lw=2.5, color="C0", label="Understood p_U")
    ax.plot(X_plot, E_s, lw=2.5, color="C1", label="Spoken p_S")
    ax.plot(X_plot, E_sign, lw=2.5, color="C2", label="Signed p_Sign")

    ax.set_xlabel("Age (months)")
    ax.set_ylabel("Expected word count")
    ax.set_ylim(-20, n_trials + 50)
    ax.legend(loc="upper left", frameon=True)
    ax.set_title("Expected vocabulary by modality")

    if output_dir is not None and filename is not None:
        fig.savefig(os.path.join(output_dir, f"{filename}.png"), dpi=300)
        fig.savefig(os.path.join(output_dir, f"{filename}.svg"))
        _save_csv(
            pd.DataFrame(
                {
                    "age_months": X_plot,
                    "understood_median": E_u,
                    "spoken_median": E_s,
                    "signed_median": E_sign,
                    "any_median": E_any,
                    "any_hdi_lo": any_hdi[:, 0],
                    "any_hdi_hi": any_hdi[:, 1],
                }
            ),
            output_dir,
            filename,
        )

    return fig


def plot_production_rate(
    samples: TrivariateModelSamples,
    hdi_prob: float = 0.90,
    output_dir: str | None = None,
    filename: str | None = None,
):
    """Plot the posterior of the production ratio q(a) = p_S(a) / p_U(a) over age."""
    X_plot = samples.X_plot
    q_plot = samples.q_plot

    q_median = np.median(q_plot, axis=1)
    q_hdi = az.hdi(q_plot, prob=hdi_prob)
    q_hdi_75 = az.hdi(q_plot, prob=0.75)
    q_hdi_50 = az.hdi(q_plot, prob=0.50)

    fig, ax = plt.subplots(figsize=plot_styles.FIGSIZE_XL)

    ax.fill_between(
        X_plot, q_hdi[:, 0], q_hdi[:, 1], alpha=0.20, label=f"{int(hdi_prob * 100)}% HDI"
    )
    ax.fill_between(X_plot, q_hdi_75[:, 0], q_hdi_75[:, 1], alpha=0.25, label="75% HDI")
    ax.fill_between(X_plot, q_hdi_50[:, 0], q_hdi_50[:, 1], alpha=0.30, label="50% HDI")
    ax.plot(X_plot, q_median, lw=3, label="Median q(a)")

    ax.set_xlabel("Age (months)")
    ax.set_ylabel("q(a) = p_S(a) / p_U(a)")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left", frameon=True)
    ax.set_title("Production ratio q(a)")

    if output_dir is not None and filename is not None:
        fig.savefig(os.path.join(output_dir, f"{filename}.png"), dpi=300)
        fig.savefig(os.path.join(output_dir, f"{filename}.svg"))
        _save_csv(
            pd.DataFrame(
                {
                    "age_months": X_plot,
                    "q_median": q_median,
                    "hdi_lo": q_hdi[:, 0],
                    "hdi_hi": q_hdi[:, 1],
                    "hdi75_lo": q_hdi_75[:, 0],
                    "hdi75_hi": q_hdi_75[:, 1],
                    "hdi50_lo": q_hdi_50[:, 0],
                    "hdi50_hi": q_hdi_50[:, 1],
                }
            ),
            output_dir,
            filename,
        )

    return fig


def plot_comprehension_production_gap(
    samples: TrivariateModelSamples,
    n_trials: int,
    hdi_prob: float = 0.90,
    output_dir: str | None = None,
    filename: str | None = None,
):
    """Plot the posterior of the comprehension-production gap (p_U - p_S) over age."""
    X_plot = samples.X_plot
    gap = (samples.p_u_plot - samples.p_s_plot) * n_trials  # in word count units

    gap_median = np.median(gap, axis=1)
    gap_hdi = az.hdi(gap, prob=hdi_prob)
    gap_hdi_50 = az.hdi(gap, prob=0.50)

    fig, ax = plt.subplots(figsize=plot_styles.FIGSIZE_XL)

    ax.fill_between(
        X_plot,
        gap_hdi[:, 0],
        gap_hdi[:, 1],
        alpha=0.20,
        label=f"{int(hdi_prob * 100)}% HDI",
    )
    ax.fill_between(X_plot, gap_hdi_50[:, 0], gap_hdi_50[:, 1], alpha=0.30, label="50% HDI")
    ax.plot(X_plot, gap_median, lw=3, label="Median gap")

    ax.set_xlabel("Age (months)")
    ax.set_ylabel("E[understood] - E[spoken] (words)")
    ax.legend(loc="upper left", frameon=True)
    ax.set_title("Comprehension-production gap")

    if output_dir is not None and filename is not None:
        fig.savefig(os.path.join(output_dir, f"{filename}.png"), dpi=300)
        fig.savefig(os.path.join(output_dir, f"{filename}.svg"))
        _save_csv(
            pd.DataFrame(
                {
                    "age_months": X_plot,
                    "gap_median": gap_median,
                    "hdi_lo": gap_hdi[:, 0],
                    "hdi_hi": gap_hdi[:, 1],
                    "hdi50_lo": gap_hdi_50[:, 0],
                    "hdi50_hi": gap_hdi_50[:, 1],
                }
            ),
            output_dir,
            filename,
        )

    return fig


# ============================================================
# Shared trivariate plotting pipeline
# ============================================================


def _run_trivariate_outcome_plots(
    samples: TrivariateModelSamples,
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
    """Run the standard per-outcome plotting pipeline for a trivariate outcome."""
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


def plot_p_any_validation(
    samples: TrivariateModelSamples,
    output_dir: str | None = None,
    filename: str | None = None,
    window: tuple[float, float] = (20.0, 56.0),
):
    """Validate the independence-based p_any against uk_02's observed union.

    p_any assumes sign and speech are conditionally independent given age. uk_02
    is the only source with the four-cell breakdown (sign-only / sign+speech /
    speech-only / understood-only), so we can compute the *observed* fraction of
    understood words produced in any modality and compare it with the model's
    union p_any / p_U over the overlap window. The model union systematically
    exceeds the observed union: the sign-speech association is positive, so
    independence over-states the total. VG15 (uk_02 multinomial) identifies it.
    """
    csv_path = os.path.join(local_env.DATA_DIR, "vocab_data_uk_02.csv")
    if not os.path.exists(csv_path):
        return None

    raw = pd.read_csv(csv_path)
    raw = raw[raw["comprehension"] > 0].copy()
    raw["union_obs"] = (
        raw["signed_only"] + raw["spoken_only"] + raw["signed_spoken"]
    ) / raw["comprehension"]
    # uk_02's own sign/speech fractions and the independence union built from
    # them. Comparing this to the observed union isolates the conditional-
    # independence bias (positive sign-speech association) from any model-vs-
    # uk_02 fit difference.
    raw["r_obs"] = (raw["signed_only"] + raw["signed_spoken"]) / raw["comprehension"]
    raw["q_obs"] = (raw["spoken_only"] + raw["signed_spoken"]) / raw["comprehension"]
    raw["indep_union"] = 1 - (1 - raw["r_obs"]) * (1 - raw["q_obs"])
    lo, hi = window
    raw = raw[(raw["age"] >= lo) & (raw["age"] <= hi)]

    # Model union fraction p_any / p_U over the same window
    X = samples.X_plot
    mask = (X >= lo) & (X <= hi)
    union_draws = samples.p_any_plot[mask] / samples.p_u_plot[mask]
    Xw = X[mask]
    union_med = np.median(union_draws, axis=1)
    union_hdi = az.hdi(union_draws, prob=0.90)

    # Binned observed union (4-month bins, >= 3 children per bin)
    edges = np.arange(lo, hi + 4, 4)
    centers, obs_means, indep_means = [], [], []
    for j in range(len(edges) - 1):
        m = (raw["age"] >= edges[j]) & (raw["age"] < edges[j + 1])
        if int(m.sum()) >= 3:
            centers.append((edges[j] + edges[j + 1]) / 2)
            obs_means.append(float(raw.loc[m, "union_obs"].mean()))
            indep_means.append(float(raw.loc[m, "indep_union"].mean()))

    fig, ax = plt.subplots(figsize=plot_styles.FIGSIZE_XL)
    ax.fill_between(
        Xw, union_hdi[:, 0], union_hdi[:, 1], alpha=0.20, color="C3",
        label="VG14 p_any / p_U (90% HDI, independence)",
    )
    ax.plot(Xw, union_med, lw=3, color="C3", label="VG14 p_any / p_U (median)")
    ax.scatter(
        raw["age"], raw["union_obs"], s=14, alpha=0.25, color="C2",
        label="uk_02 observed union (per child)",
    )
    if centers:
        ax.plot(
            centers, obs_means, "o-", color="C2", lw=2, ms=7,
            label="uk_02 observed union (binned mean)",
        )
        ax.plot(
            centers, indep_means, "s--", color="C0", lw=2, ms=6,
            label="uk_02 independence union 1-(1-r)(1-q)",
        )
    ax.set_xlabel("Age (months)")
    ax.set_ylabel("Union fraction of understood words produced")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left", frameon=True)
    ax.set_title("p_any validation: independence over-states the observed union")

    obs_mean = float(raw["union_obs"].mean())
    indep_mean = float(raw["indep_union"].mean())
    mod_mean = float(np.mean(union_med))
    # Pure independence bias, computed within uk_02 (its own r,q): isolates the
    # positive sign-speech association from model fit.
    indep_bias_pp = 100.0 * (indep_mean - obs_mean)
    # Model check: VG14's independence union vs the observed union (conflates the
    # association with the model's r,q differing from uk_02's in-sample r,q).
    model_gap_pp = 100.0 * (mod_mean - obs_mean)

    if output_dir is not None and filename is not None:
        fig.savefig(os.path.join(output_dir, f"{filename}.png"), dpi=300)
        fig.savefig(os.path.join(output_dir, f"{filename}.svg"))
        _save_csv(
            pd.DataFrame(
                {
                    "age_months": Xw,
                    "model_union_median": union_med,
                    "model_union_hdi_lo": union_hdi[:, 0],
                    "model_union_hdi_hi": union_hdi[:, 1],
                }
            ),
            output_dir,
            filename,
        )
        _save_csv(
            pd.DataFrame(
                {
                    "window_lo": [lo],
                    "window_hi": [hi],
                    "uk02_observed_union_mean": [obs_mean],
                    "uk02_independence_union_mean": [indep_mean],
                    "independence_bias_pp": [indep_bias_pp],
                    "model_union_mean": [mod_mean],
                    "model_gap_pp": [model_gap_pp],
                }
            ),
            output_dir,
            f"{filename}_gap",
        )

    heading("p_any validation against uk_02 four-cell union", style="bold cyan")
    key_value_table(
        f"Union of understood words produced (ages {lo:.0f}-{hi:.0f} mo)",
        [
            ("uk_02 observed union (mean)", round(obs_mean, 3)),
            ("uk_02 independence union (mean)", round(indep_mean, 3)),
            ("Independence bias within uk_02 (pp)", round(indep_bias_pp, 1)),
            ("VG14 p_any / p_U (mean)", round(mod_mean, 3)),
            ("VG14 vs observed union (pp)", round(model_gap_pp, 1)),
        ],
    )

    return fig


def _run_trivariate_plots(context: TrivariateContext):
    """Run the joint trivariate plots and per-outcome plots for VG14."""
    samples = context.model_samples
    analysis_df = context.analysis_df
    has_u = analysis_df["understood"].notna()
    has_s = analysis_df["spoken"].notna()
    has_sign = analysis_df["signed"].notna()
    n_trials = context.model_data.n_trials
    hdi_prob = context.reporting.hdi
    output_dir = context.reporting.output_dir

    # ---- Joint trajectory (understood, spoken, signed) ----
    fig = plot_understood_spoken_signed_trajectory(
        samples, n_trials=n_trials, output_dir=output_dir, filename="joint_trajectory"
    )
    context.plots["joint_trajectory"] = fig
    plt.close(fig)

    # ---- Modality trajectories (p_U, p_S, p_Sign, p_any) ----
    fig = plot_modality_trajectories(
        samples,
        n_trials=n_trials,
        hdi_prob=hdi_prob,
        output_dir=output_dir,
        filename="modality_trajectories",
    )
    context.plots["modality_trajectories"] = fig
    plt.close(fig)

    # ---- Production rate q(a) ----
    fig = plot_production_rate(
        samples, hdi_prob=hdi_prob, output_dir=output_dir, filename="production_rate"
    )
    context.plots["production_rate"] = fig
    plt.close(fig)

    # ---- Signed rate r(a) (extrapolation outside signing support shaded) ----
    fig = plot_signed_rate(
        samples,
        hdi_prob=hdi_prob,
        output_dir=output_dir,
        filename="signed_rate",
        support_range=SIGNED_SUPPORT_MONTHS,
    )
    context.plots["signed_rate"] = fig
    plt.close(fig)

    # ---- Sign -> speech crossover ----
    fig = plot_sign_speech_crossover(
        samples,
        hdi_prob=hdi_prob,
        output_dir=output_dir,
        filename="sign_speech_crossover",
        support_range=SIGNED_SUPPORT_MONTHS,
    )
    context.plots["sign_speech_crossover"] = fig
    plt.close(fig)

    # ---- p_any validation against uk_02 four-cell union ----
    fig = plot_p_any_validation(
        samples, output_dir=output_dir, filename="p_any_validation"
    )
    if fig is not None:
        context.plots["p_any_validation"] = fig
        plt.close(fig)

    # ---- Comprehension-production gap ----
    fig = plot_comprehension_production_gap(
        samples,
        n_trials=n_trials,
        hdi_prob=hdi_prob,
        output_dir=output_dir,
        filename="comprehension_production_gap",
    )
    context.plots["comprehension_production_gap"] = fig
    plt.close(fig)

    # ---- Per-outcome plots: understood ----
    _run_trivariate_outcome_plots(
        samples=samples,
        y_plot=samples.y_u_plot,
        y_query=samples.y_u_query,
        f_plot=samples.f_u_plot,
        kappa_plot=samples.kappa_u_plot,
        kappa_query=samples.kappa_u_query,
        x_obs=analysis_df.loc[has_u, "age"],
        y_obs=analysis_df.loc[has_u, "understood"],
        n_trials=n_trials,
        hdi_prob=hdi_prob,
        output_dir=output_dir,
        suffix="u",
        outcome_label="Words understood",
        y_label="Predicted words understood",
    )

    # ---- Per-outcome plots: spoken ----
    _run_trivariate_outcome_plots(
        samples=samples,
        y_plot=samples.y_s_plot,
        y_query=samples.y_s_query,
        f_plot=samples.f_s_plot,
        kappa_plot=samples.kappa_s_plot,
        kappa_query=samples.kappa_s_query,
        x_obs=analysis_df.loc[has_s, "age"],
        y_obs=analysis_df.loc[has_s, "spoken"],
        n_trials=n_trials,
        hdi_prob=hdi_prob,
        output_dir=output_dir,
        suffix="s",
        outcome_label="Words spoken",
        y_label="Predicted words spoken",
    )

    # ---- Per-outcome plots: signed ----
    _run_trivariate_outcome_plots(
        samples=samples,
        y_plot=samples.y_sign_plot,
        y_query=samples.y_sign_query,
        f_plot=samples.f_sign_plot,
        kappa_plot=samples.kappa_sign_plot,
        kappa_query=samples.kappa_sign_query,
        x_obs=analysis_df.loc[has_sign, "age"],
        y_obs=analysis_df.loc[has_sign, "signed"],
        n_trials=n_trials,
        hdi_prob=hdi_prob,
        output_dir=output_dir,
        suffix="sign",
        outcome_label="Words signed",
        y_label="Predicted words signed",
    )


# ============================================================
# Fit orchestration
# ============================================================


def fit_trivariate_model(
    config: str,
    definition: TrivariateModelDefinition,
) -> TrivariateContext:
    """
    Shared fit pipeline for the trivariate model (VG14).
    """
    run_banner(definition.banner, subtitle=f"sampling config: {config}")

    env_info.report_environment_info()

    console.print()
    package_metadata.report_package_versions(PACKAGE_LIST)

    context: TrivariateContext = ModelFitContext(
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
        prepare_trivariate_data(context, definition)

    with section("Priors and hyperparameters", timings=timings):
        configure_trivariate_priors(context, definition)

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
        _run_trivariate_plots(context)

    with section("Report", timings=timings):
        report(context)

    pipeline_summary(f"Pipeline summary — {context.reporting.model_label}", timings)
    console.print(
        f"[dim]Total wall time: "
        f"{vg_reporting.format_duration(time.perf_counter() - run_started)}[/dim]"
    )

    return context
