# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Bivariate vocabulary growth model with study-level random intercepts (VG07).

Extends the production-ratio reparameterization from common_bivariate with
study-level random intercepts on both the understood trajectory and the
production ratio:

    f_U(a, s) = mean_trend_u(a) + g_u(a) + delta_u[s]
    h(a, s)   = mean_trend_q(a) + g_q(a) + delta_q[s]

    delta_u[s] ~ Normal(0, tau_u)
    delta_q[s] ~ Normal(0, tau_q)

Plot and query predictions use the population-level trajectory (delta=0).
"""

import os
import shutil

import dse_research_utils.environment.info as env_info
import dse_research_utils.math.constants as math_constants
import dse_research_utils.metadata.packages as package_metadata
import dse_research_utils.statistics.models.data as model_data
import dse_research_utils.statistics.models.pymc_utils as pymc_utils
import dse_research_utils.statistics.models.reporting as reporting
import dse_research_utils.statistics.models.sampling as sampling
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
from rich import print

import vocab_growth.data_utils as vocab_data_utils
import vocab_growth.environment as local_env
import vocab_growth.plotting as plotting
from vocab_growth.models.common import (
    PACKAGE_LIST,
    ModelFitContext,
    get_hsgp_hyperparams,
    report,
)
from vocab_growth.models.common_bivariate import (
    BivariateContext,
    configure_bivariate_priors,
    diagnostics,
    plot_comprehension_production_gap,
    plot_production_rate,
    plot_production_rate_by_understood,
    plot_production_rate_predictive,
    plot_understood_spoken_trajectory,
    plot_understood_spoken_trajectory_hdi,
    plot_understood_vs_spoken,
    plot_understood_vs_spoken_predictive,
    posterior_summary,
    prior_predictive_checks,
    sample,
    sample_posterior_predictive,
)
from vocab_growth.models.definitions import BivariateModelDefinition

EPSILON = math_constants.EPSILON

BivariateREContext = BivariateContext


# ============================================================
# Data preparation (with study column)
# ============================================================


def prepare_bivariate_re_data(
    context: BivariateREContext,
    definition: BivariateModelDefinition,
):
    """Load and prepare data for a bivariate model with study random effects."""
    print(
        "\n[green]------------------------------------------------------------[/green]"
    )
    print("[bold green]Prepare data[/bold green]")
    print("[green]------------------------------------------------------------[/green]")
    print()

    import dse_research_utils.statistics.descriptive as descriptive_stats
    from rich.pretty import pprint

    df = vocab_data_utils.load_data(
        population=definition.population,
        columns=["age", "understood", "spoken", "study"],
        sample_fraction=definition.sample_fraction,
        random_seed=definition.random_seed,
    )
    analysis_df = df[["age", "understood", "spoken", "study"]].copy()

    # Keep rows where at least one outcome is observed (and age is present)
    analysis_df = analysis_df.dropna(subset=["age"])
    has_u = analysis_df["understood"].notna()
    has_s = analysis_df["spoken"].notna()
    analysis_df = analysis_df[has_u | has_s].reset_index(drop=True)

    # Create integer study codes
    unique_studies = sorted(analysis_df["study"].unique())
    study_map = {s: i for i, s in enumerate(unique_studies)}
    analysis_df["study_code"] = analysis_df["study"].map(study_map).astype(int)

    desc = descriptive_stats.describe_all(
        analysis_df[["age", "understood", "spoken"]], alpha=0.05
    )

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
    n_studies = len(unique_studies)

    print(f"\n  Total observations:       {n}")
    print(f"  Understood observed:      {n_u}")
    print(f"  Spoken observed:          {n_s}")
    print(f"  Both observed:            {n_both}")
    print(f"  Understood only:          {n_u - n_both}")
    print(f"  Spoken only:              {n_s - n_both}")
    print(f"  Studies:                  {n_studies} {unique_studies}")

    # Create a BinomialModelData for the context interface
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
# Model building (with study random intercepts)
# ============================================================


def build_model_re(
    context: BivariateREContext,
    definition: BivariateModelDefinition,
):
    """Build the bivariate PyMC model with study-level random intercepts."""
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
    study_codes = np.asarray(analysis_df["study_code"], dtype=int)

    idx_u = np.where(has_u)[0]
    idx_s = np.where(has_s)[0]

    n = len(X_obs)
    n_u = len(y_u_observed)
    n_s = len(y_s_observed)
    n_trials = context.model_data.n_trials
    n_studies = int(study_codes.max()) + 1

    print(f"  Total observations:   {n}")
    print(f"  Understood observed:  {n_u}")
    print(f"  Spoken observed:      {n_s}")
    print(f"  n_trials:             {n_trials}")
    print(f"  n_studies:            {n_studies}")

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
        "study_id": np.arange(n_studies),
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

        study_obs = pm.Data("study_obs", study_codes, dims=("obs_id",))

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

        # Population-level f_U (no study effect)
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

        # Population-level h (no study effect)
        h_all = pm.Deterministic("h_all", mean_trend_q + g_q, dims=("all_id",))

        # ============================================================
        # Study-level random intercepts
        # ============================================================

        tau_u = pm.HalfNormal("tau_u", sigma=definition.tau_u_sigma)
        delta_u = pm.Normal("delta_u", mu=0, sigma=tau_u, dims="study_id")

        tau_q = pm.HalfNormal("tau_q", sigma=definition.tau_q_sigma)
        delta_q = pm.Normal("delta_q", mu=0, sigma=tau_q, dims="study_id")

        # ============================================================
        # Observation-level quantities (with study effects)
        # ============================================================

        # Understood — obs level includes study shift
        f_u_obs_re = f_u_all[i_obs0:i_obs1] + delta_u[study_obs]
        p_u_obs = pm.Deterministic(
            "p_u_obs", pm.math.sigmoid(f_u_obs_re), dims=("obs_id",)
        )
        # For diagnostics: population-level f at obs ages
        _ = pm.Deterministic("f_u_obs", f_u_all[i_obs0:i_obs1], dims=("obs_id",))

        # Production ratio — obs level includes study shift
        h_obs_re = h_all[i_obs0:i_obs1] + delta_q[study_obs]
        q_obs = pm.Deterministic(
            "q_obs", pm.math.sigmoid(h_obs_re), dims=("obs_id",)
        )
        _ = pm.Deterministic("h_obs", h_all[i_obs0:i_obs1], dims=("obs_id",))

        # Spoken — derived from obs-level p_U and q (with study effects)
        p_s_obs = pm.Deterministic("p_s_obs", p_u_obs * q_obs, dims=("obs_id",))

        p_s_obs_clip = pm.math.clip(p_s_obs, EPSILON, 1 - EPSILON)
        _ = pm.Deterministic(
            "f_s_obs",
            pm.math.log(p_s_obs_clip) - pm.math.log(1 - p_s_obs_clip),
            dims=("obs_id",),
        )

        # ============================================================
        # Population-level quantities (no study effect) — plot/query
        # ============================================================

        _ = pm.Deterministic("f_u_plot", f_u_all[i_plot0:i_plot1], dims=("plot_id",))
        _ = pm.Deterministic(
            "f_u_query", f_u_all[i_query0:i_query1], dims=("query_id",)
        )

        p_u_all = pm.math.sigmoid(f_u_all)
        _ = pm.Deterministic(
            "p_u_plot", p_u_all[i_plot0:i_plot1], dims=("plot_id",)
        )
        _ = pm.Deterministic(
            "p_u_query", p_u_all[i_query0:i_query1], dims=("query_id",)
        )

        _ = pm.Deterministic("h_plot", h_all[i_plot0:i_plot1], dims=("plot_id",))
        _ = pm.Deterministic("h_query", h_all[i_query0:i_query1], dims=("query_id",))

        q_all = pm.math.sigmoid(h_all)
        _ = pm.Deterministic(
            "q_plot", q_all[i_plot0:i_plot1], dims=("plot_id",)
        )
        _ = pm.Deterministic(
            "q_query", q_all[i_query0:i_query1], dims=("query_id",)
        )

        p_s_all = p_u_all * q_all
        _ = pm.Deterministic(
            "p_s_plot", p_s_all[i_plot0:i_plot1], dims=("plot_id",)
        )
        _ = pm.Deterministic(
            "p_s_query", p_s_all[i_query0:i_query1], dims=("query_id",)
        )

        p_s_all_clip = pm.math.clip(p_s_all, EPSILON, 1 - EPSILON)
        f_s_all = pm.math.log(p_s_all_clip) - pm.math.log(1 - p_s_all_clip)
        _ = pm.Deterministic(
            "f_s_plot", f_s_all[i_plot0:i_plot1], dims=("plot_id",)
        )
        _ = pm.Deterministic(
            "f_s_query", f_s_all[i_query0:i_query1], dims=("query_id",)
        )

        # Standardised ages
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
# Fit orchestration
# ============================================================


def fit_bivariate_re_model(
    config: str,
    definition: BivariateModelDefinition,
) -> BivariateREContext:
    """
    Fit pipeline for bivariate model with study random intercepts (VG07).
    """
    print(
        "\n[green]============================================================[/green]"
    )
    print(
        f"[bold green]{definition.banner}[/bold green]"
    )
    print("[green]============================================================[/green]")
    print()

    env_info.report_environment_info()

    print()
    package_metadata.report_package_versions(PACKAGE_LIST)

    context: BivariateREContext = ModelFitContext(
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

    prepare_bivariate_re_data(context, definition)

    configure_bivariate_priors(context, definition)

    build_model_re(context, definition)

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
