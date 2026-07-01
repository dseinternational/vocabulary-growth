# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Univariate vocabulary growth model with study-level random intercepts.

Extends the single-outcome HSGP model from common.py with dataset-level
random intercepts on the outcome trajectory:

    f(a, s) = mean_trend(a) + g(a) + delta[s]

    delta[s] ~ Normal(0, tau)
    tau ~ HalfNormal(tau_study_sigma)

Plot and query predictions use the population-level trajectory (delta = 0).

This module is the shared pipeline for VG11 (TD spoken) and VG12 (TD understood).
"""

import os
import shutil
import time

import dse_research_utils.environment.info as env_info
import dse_research_utils.math.constants as math_constants
import dse_research_utils.metadata.packages as package_metadata
import dse_research_utils.statistics.descriptive as descriptive_stats
import dse_research_utils.statistics.models.data as model_data
import dse_research_utils.statistics.models.pymc_utils as pymc_utils
import dse_research_utils.statistics.models.reporting as reporting
import dse_research_utils.statistics.models.sampling as sampling
import numpy as np
import pymc as pm

import vocab_growth.data_utils as vocab_data_utils
import vocab_growth.environment as local_env
import vocab_growth.reporting as vg_reporting
from vocab_growth.models.build_utils import (
    construct_age_grids,
    slope_anchor_logit_coeffs,
    standardize_ages,
    validate_ell_bounds,
)
from vocab_growth.models.common import (
    PACKAGE_LIST,
    ModelFitContext,
    configure_univariate_priors,
    diagnostics,
    get_hsgp_hyperparams,
    posterior_summary,
    prior_predictive_checks,
    render_model_graph,
    report,
    run_standard_plots,
    sample,
    sample_posterior_predictive,
)
from vocab_growth.models.definitions import UnivariateModelDefinition
from vocab_growth.models.gp_utils import GPGrid, make_kappa_of_z, trend_and_gp
from vocab_growth.reporting import (
    console,
    dataframe_table,
    key_value_table,
    pipeline_summary,
    run_banner,
    section,
)

EPSILON = math_constants.EPSILON

# Type alias — the RE variant uses the same context type as the base model.
UnivariateREContext = ModelFitContext


# ============================================================
# Data preparation (with study column)
# ============================================================


def prepare_univariate_re_data(
    context: UnivariateREContext,
    definition: UnivariateModelDefinition,
) -> None:
    """Load and prepare data for a univariate model with study random effects.

    For DS data the ``study`` column comes directly from ``vocab_combined``.
    For TD data the ``dataset_name`` column from Wordbank is used as the study
    grouping, since it captures genuine between-lab variation that is more
    meaningful than CDI form (WG / WS).
    """
    y_col = definition.outcome.value
    columns = ["age", y_col, "study"]

    df = vocab_data_utils.load_data(
        population=definition.population,
        columns=columns,
        sample_fraction=definition.sample_fraction,
        random_seed=definition.random_seed,
    )
    analysis_df = df[columns].dropna(subset=["age", y_col]).reset_index(drop=True)
    analysis_df, dropped_studies = vocab_data_utils.filter_studies_by_min_obs(
        analysis_df, definition.min_study_observations
    )

    # Integer study codes (sorted for reproducibility)
    unique_studies = sorted(analysis_df["study"].unique())
    study_map = {s: i for i, s in enumerate(unique_studies)}
    analysis_df["study_code"] = analysis_df["study"].map(study_map).astype(int)
    n_studies = len(unique_studies)

    desc = descriptive_stats.describe_all(analysis_df[["age", y_col]], alpha=0.05)

    # Per-study observation counts for diagnostics
    study_counts = (
        analysis_df.groupby("study")
        .size()
        .reset_index(name="n")
        .sort_values("study")
    )

    data_rows = [
        ("Population", definition.population.name),
        ("Outcome column", y_col),
        ("Total observations", len(analysis_df)),
        ("Sample fraction", definition.sample_fraction),
        ("Studies", f"{n_studies} ({', '.join(map(str, unique_studies))})"),
    ]
    if definition.min_study_observations:
        data_rows.append(
            (
                f"Studies dropped (<{definition.min_study_observations} obs)",
                ", ".join(dropped_studies) if dropped_studies else "none",
            )
        )
    key_value_table("Data", data_rows)
    dataframe_table(study_counts, title="Observations per study")
    dataframe_table(desc, title="Descriptive statistics")

    X_obs = np.asarray(analysis_df["age"], dtype=float).reshape(-1, 1)
    y_obs = np.asarray(analysis_df[y_col], dtype=int)

    bmd = model_data.BinomialModelData(
        X_obs=X_obs, y_obs=y_obs, n_trials=definition.n_trials
    )

    context.set_model_data(bmd, analysis_df)
    context.dataframes["descriptive_stats"] = desc
    context.dataframes["study_counts"] = study_counts

    desc.to_csv(
        os.path.join(context.reporting.output_dir, "descriptive_statistics.csv"),
        index=True,
    )
    study_counts.to_csv(
        os.path.join(context.reporting.output_dir, "study_counts.csv"),
        index=False,
    )


# ============================================================
# Model building (with study random intercepts)
# ============================================================


def build_univariate_re_model(
    context: UnivariateREContext,
    definition: UnivariateModelDefinition,
) -> None:
    """Build the univariate PyMC model with study-level random intercepts.

    The study intercept ``delta[s]`` shifts the population-level linear
    predictor ``f(a)`` at the observation level only; plot and query
    predictions use the population-level trajectory (``delta = 0``).

    When ``definition.anchor_g_at_ref`` is True the GP is reparameterised so
    that every posterior draw passes through zero at ``gp_anchor_age_months``
    (defaulting to the midpoint of ``slope_anchors``).  This removes the
    per-draw ridge between the global intercept and a constant GP component,
    which is especially important when study intercepts are also present.
    """
    config = context.model_config
    analysis_df = context.analysis_df

    y_col = definition.outcome.value

    X_obs = np.asarray(analysis_df["age"], dtype=float).reshape(-1, 1)
    y_obs = np.asarray(analysis_df[y_col], dtype=int)
    study_codes = np.asarray(analysis_df["study_code"], dtype=int)

    n = len(X_obs)
    n_trials = context.model_data.n_trials
    n_studies = int(study_codes.max()) + 1

    # Validate
    if not np.all(y_obs >= 0):
        raise ValueError("y_obs contains negative counts.")
    if not np.all(y_obs <= n_trials):
        raise ValueError("y_obs exceeds n_trials.")

    # Standardise ages
    X_obs_mean, X_obs_std, X_obs_z = standardize_ages(X_obs)

    key_value_table(
        "Build configuration",
        [
            ("Number of observations", n),
            ("Number of trials (n_trials)", n_trials),
            ("Number of studies", n_studies),
            ("Slope anchors (months)", config.slope_anchors),
            ("Length-scale range (months)", config.ell_months_range),
            ("Number of plot points", config.n_plot),
            ("Query ages (months)", config.ages_query),
            ("Age median (months)", float(np.median(X_obs))),
            ("Age mean (months)", X_obs_mean),
            ("Age std (months)", X_obs_std),
        ],
    )

    # Plot / query grids (standardised), with the optional reference-age anchor
    # row — see models.build_utils.construct_age_grids.
    anchor_g = bool(definition.anchor_g_at_ref)
    grids = construct_age_grids(
        X_obs,
        X_obs_z,
        X_obs_mean=X_obs_mean,
        X_obs_std=X_obs_std,
        n_plot=config.n_plot,
        ages_query=config.ages_query,
        slope_anchors=config.slope_anchors,
        use_gp_anchor=anchor_g,
        gp_anchor_age_months=definition.gp_anchor_age_months,
    )
    X_plot = grids.X_plot
    X_query = grids.X_query
    X_all_z = grids.X_all_z
    n_plot = grids.n_plot
    n_query = grids.n_query
    n_all = grids.n_all
    i_anchor = grids.i_anchor
    anchor_age_months = grids.anchor_age_months

    # Length-scale bounds
    ell_low_months, ell_high_months = validate_ell_bounds(config.ell_months_range)
    ell_low_z = ell_low_months / X_obs_std
    ell_high_z = ell_high_months / X_obs_std
    ell_range_z = (ell_low_z, ell_high_z)

    L, M = get_hsgp_hyperparams(X_obs_z, ell_range_z)

    # Slope anchors
    slope_age_a_z, slope_age_b_z = slope_anchor_logit_coeffs(
        config.slope_anchors, X_obs_mean=X_obs_mean, X_obs_std=X_obs_std
    )

    derived_rows: list[tuple[str, object]] = [
        ("HSGP basis size (m)", M),
        ("HSGP boundary factor (L)", L),
        ("Slope anchors (z-score)", (slope_age_a_z, slope_age_b_z)),
        ("Length-scale range (z-score)", (ell_low_z, ell_high_z)),
    ]
    if anchor_g:
        derived_rows.append(("GP anchor age (months)", f"{anchor_age_months:g}"))
    key_value_table("Derived quantities", derived_rows)

    # Slice indices
    i_obs0, i_obs1 = 0, n
    i_plot0, i_plot1 = i_obs1, i_obs1 + n_plot
    i_query0, i_query1 = i_plot1, i_plot1 + n_query

    coords = {
        "all_id": np.arange(n_all),
        "obs_id": np.arange(n),
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

        study_obs = pm.Data("study_obs", study_codes, dims=("obs_id",))

        # ============================================================
        # Mean developmental trajectory + HSGP deviation
        # ============================================================
        # Shared helper (gp_utils); graph byte-identical to the inlined form
        # (stores the named g / f_all; Option-D anchor when the model enables it).
        # Population-level linear predictor (no study effect).
        f_all = trend_and_gp(
            cfg_low=config.p_slope_low_dist,
            cfg_hi=config.p_slope_hi_dist,
            cfg_ell=config.ell_unit_dist,
            cfg_eta=config.eta_dist,
            suffix="",
            X_all_z_data=X_all_z_data,
            grid=GPGrid(
                sa_z=slope_age_a_z,
                sb_z=slope_age_b_z,
                ell_low_z=ell_low_z,
                ell_high_z=ell_high_z,
                M=M,
                L=L,
            ),
            store_deterministic=True,
            latent_name="f_all",
            anchor_idx=i_anchor if anchor_g else None,
        )

        # ============================================================
        # Study-level random intercepts (non-centred parameterisation)
        # ============================================================

        tau = pm.HalfNormal("tau", sigma=definition.tau_study_sigma)
        delta_raw = pm.Normal("delta_raw", mu=0.0, sigma=1.0, dims="study_id")
        delta = pm.Deterministic("delta", tau * delta_raw, dims="study_id")

        # ============================================================
        # Observation-level quantities (with study shift)
        # ============================================================

        # Add study shift to obs-level predictor only
        f_obs_re = f_all[i_obs0:i_obs1] + delta[study_obs]

        f_obs = pm.Deterministic("f_obs", f_obs_re, dims=("obs_id",))
        p_obs = pm.Deterministic("p_obs", pm.math.sigmoid(f_obs), dims=("obs_id",))

        # ============================================================
        # Population-level quantities (no study effect) — plot/query
        # ============================================================

        f_plot = pm.Deterministic("f_plot", f_all[i_plot0:i_plot1], dims=("plot_id",))
        f_query = pm.Deterministic(
            "f_query", f_all[i_query0:i_query1], dims=("query_id",)
        )

        _ = pm.Deterministic("p_plot", pm.math.sigmoid(f_plot), dims=("plot_id",))
        _ = pm.Deterministic("p_query", pm.math.sigmoid(f_query), dims=("query_id",))

        # Standardised ages (needed by extract_model_samples)
        _ = pm.Deterministic(
            "z_obs", X_all_z_data[i_obs0:i_obs1, 0], dims=("obs_id",)
        )
        _ = pm.Deterministic(
            "z_plot", X_all_z_data[i_plot0:i_plot1, 0], dims=("plot_id",)
        )
        _ = pm.Deterministic(
            "z_query", X_all_z_data[i_query0:i_query1, 0], dims=("query_id",)
        )

        # ============================================================
        # Dispersion / overdispersion
        # ============================================================

        kappa_min = config.kappa_min_dist.to_pymc("kappa_min")
        a_kappa = config.a_kappa_dist.to_pymc("a_kappa")
        b_kappa_mag = config.b_kappa_mag_dist.to_pymc("b_kappa_mag")
        b_kappa = pm.Deterministic("b_kappa", -b_kappa_mag)

        kappa_of_z = make_kappa_of_z(kappa_min, a_kappa, b_kappa)

        kappa_obs = pm.Deterministic(
            "kappa_obs", kappa_of_z(X_all_z_data[i_obs0:i_obs1, 0]), dims="obs_id"
        )
        _ = pm.Deterministic(
            "kappa_plot", kappa_of_z(X_all_z_data[i_plot0:i_plot1, 0]), dims="plot_id"
        )
        _ = pm.Deterministic(
            "kappa_query", kappa_of_z(X_all_z_data[i_query0:i_query1, 0]), dims="query_id"
        )

        # ============================================================
        # Beta-Binomial likelihood
        # ============================================================

        p_obs_clip = pm.math.clip(p_obs, EPSILON, 1 - EPSILON)
        alpha_obs = p_obs_clip * kappa_obs
        beta_obs = (1 - p_obs_clip) * kappa_obs

        _ = pm.BetaBinomial(
            "y_obs",
            n=n_trials,
            alpha=alpha_obs,
            beta=beta_obs,
            observed=y_obs,
            dims=("obs_id",),
        )

    variables = pymc_utils.get_variables_dict(model_pm)

    pymc_utils.report_model_summary(model_pm)

    render_model_graph(model_pm, context.reporting.output_dir)

    context.set_model(model_pm, variables)


# ============================================================
# Fit orchestration
# ============================================================


def fit_univariate_re_model(
    config: str,
    definition: UnivariateModelDefinition,
) -> UnivariateREContext:
    """Fit pipeline for a univariate model with study random intercepts.

    Mirrors ``fit_single_outcome_model`` from ``common.py`` but swaps in the
    RE-aware data-prep and model-build steps.  All downstream pipeline stages
    (prior predictive checks, sampling, diagnostics, posterior summary, plots,
    report) are reused unchanged from ``common.py``.
    """
    run_banner(definition.banner, subtitle=f"sampling config: {config}")

    env_info.report_environment_info()

    console.print()
    package_metadata.report_package_versions(PACKAGE_LIST)

    context: UnivariateREContext = ModelFitContext(
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

    y_col = definition.outcome.value
    outcome_label = definition.outcome_label

    with section("Prepare data", timings=timings):
        prepare_univariate_re_data(context, definition)

    with section("Priors and hyperparameters", timings=timings):
        configure_univariate_priors(context, definition)

    with section("Model definition and initialisation", timings=timings):
        build_univariate_re_model(context, definition)

    with section("Prior predictive checks", timings=timings):
        prior_predictive_checks(
            context, outcome_col=y_col, outcome_label=outcome_label
        )

    with section("Posterior sampling", timings=timings):
        sample(context)

    with section("Diagnostics", timings=timings):
        diagnostics(context)

    with section("Posterior predictions", timings=timings):
        sample_posterior_predictive(context)

    with section("Posterior summary", timings=timings):
        posterior_summary(context)

    with section("Plots", timings=timings):
        run_standard_plots(context, outcome_label=outcome_label)

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
