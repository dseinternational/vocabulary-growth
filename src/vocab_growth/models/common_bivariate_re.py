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
from vocab_growth.models.common import (
    PACKAGE_LIST,
    ModelFitContext,
    get_hsgp_hyperparams,
    render_model_graph,
    report,
)
from vocab_growth.models.common_bivariate import (
    BivariateContext,
    _run_bivariate_joint_plots,
    configure_bivariate_priors,
    diagnostics,
    posterior_summary,
    prior_predictive_checks,
    sample,
    sample_posterior_predictive,
)
from vocab_growth.models.definitions import BivariateModelDefinition
from vocab_growth.reporting import (
    console,
    dataframe_table,
    key_value_table,
    pipeline_summary,
    run_banner,
    section,
)

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
    columns = ["age", "understood", "spoken", "study"]
    use_subject_codes = definition.use_subject_re_u or definition.use_subject_re_q
    if use_subject_codes:
        columns = columns + ["subject_id"]

    df = vocab_data_utils.load_data(
        population=definition.population,
        columns=columns,
        sample_fraction=definition.sample_fraction,
        random_seed=definition.random_seed,
        max_age_months=definition.max_age_months,
    )
    analysis_df = df[columns].copy()

    # Keep rows where at least one outcome is observed (and age is present)
    analysis_df = analysis_df.dropna(subset=["age"])
    has_u = analysis_df["understood"].notna()
    has_s = analysis_df["spoken"].notna()
    analysis_df = analysis_df[has_u | has_s].reset_index(drop=True)
    analysis_df, dropped_studies = vocab_data_utils.filter_studies_by_min_obs(
        analysis_df, definition.min_study_observations
    )

    # Create integer study codes
    unique_studies = sorted(analysis_df["study"].unique())
    study_map = {s: i for i, s in enumerate(unique_studies)}
    analysis_df["study_code"] = analysis_df["study"].map(study_map).astype(int)

    # Create integer subject codes (unique across studies)
    n_subjects: int | None = None
    if use_subject_codes:
        subj_keys = (
            analysis_df["study"].astype(str)
            + "::"
            + analysis_df["subject_id"].astype(str)
        )
        analysis_df["subject_key"] = subj_keys
        unique_subjects = sorted(subj_keys.unique())
        subject_map = {s: i for i, s in enumerate(unique_subjects)}
        analysis_df["subject_code"] = subj_keys.map(subject_map).astype(int)
        n_subjects = len(unique_subjects)

    desc = descriptive_stats.describe_all(
        analysis_df[["age", "understood", "spoken"]], alpha=0.05
    )

    n = len(analysis_df)
    n_u = int(analysis_df["understood"].notna().sum())
    n_s = int(analysis_df["spoken"].notna().sum())
    n_both = int(
        (analysis_df["understood"].notna() & analysis_df["spoken"].notna()).sum()
    )
    n_studies = len(unique_studies)

    counts: list[tuple[str, object]] = [
        ("Total observations", n),
        ("Understood observed", n_u),
        ("Spoken observed", n_s),
        ("Both observed", n_both),
        ("Understood only", n_u - n_both),
        ("Spoken only", n_s - n_both),
        ("Studies", f"{n_studies} ({', '.join(map(str, unique_studies))})"),
    ]
    if definition.min_study_observations:
        counts.append(
            (
                f"Studies dropped (<{definition.min_study_observations} obs)",
                ", ".join(dropped_studies) if dropped_studies else "none",
            )
        )
    if n_subjects is not None:
        n_singletons = int(
            (analysis_df.groupby("subject_code").size() == 1).sum()
        )
        counts.append(("Subjects", n_subjects))
        counts.append(("Subjects with single observation", n_singletons))
        counts.append(("Subjects with repeated observations", n_subjects - n_singletons))
    key_value_table("Observation counts", counts)
    dataframe_table(desc, title="Descriptive statistics")

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

    analysis_df = context.analysis_df

    # Observation masks
    has_u = analysis_df["understood"].notna().values
    has_s = analysis_df["spoken"].notna().values

    # Optional held-out mask: rows where holdout==True remain in obs space (so
    # f_u_obs, h_obs etc. are computed at their ages) but are excluded from the
    # likelihood. Subject REs for held-out subjects are then drawn from the
    # prior, which is exactly what we need for K-fold LOSO.
    if "holdout" in analysis_df.columns:
        holdout = analysis_df["holdout"].fillna(False).astype(bool).values
    else:
        holdout = np.zeros(len(analysis_df), dtype=bool)
    has_u_train = has_u & ~holdout
    has_s_train = has_s & ~holdout

    X_obs = np.asarray(analysis_df["age"], dtype=float).reshape(-1, 1)
    y_u_observed = np.asarray(
        analysis_df.loc[has_u_train, "understood"], dtype=int
    )
    y_s_observed = np.asarray(
        analysis_df.loc[has_s_train, "spoken"], dtype=int
    )
    study_codes = np.asarray(analysis_df["study_code"], dtype=int)

    idx_u = np.where(has_u_train)[0]
    idx_s = np.where(has_s_train)[0]

    n = len(X_obs)
    n_u = len(y_u_observed)
    n_s = len(y_s_observed)
    n_trials = context.model_data.n_trials
    n_studies = int(study_codes.max()) + 1

    use_subject_re_u = bool(definition.use_subject_re_u)
    use_subject_re_q = bool(definition.use_subject_re_q)
    use_subject_codes = use_subject_re_u or use_subject_re_q
    if use_subject_codes:
        subject_codes = np.asarray(analysis_df["subject_code"], dtype=int)
        n_subjects = int(subject_codes.max()) + 1
    else:
        subject_codes = None
        n_subjects = 0

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

    build_cfg: list[tuple[str, object]] = [
        ("Total observations", n),
        ("Understood observed", n_u),
        ("Spoken observed", n_s),
        ("n_trials", n_trials),
        ("n_studies", n_studies),
    ]
    if use_subject_codes:
        build_cfg.append(("n_subjects", n_subjects))
    build_cfg.extend(
        [
            ("Age mean (months)", X_obs_mean),
            ("Age std (months)", X_obs_std),
            ("Slope anchors (months)", config.slope_anchors),
            ("Length-scale range (months)", config.ell_months_range),
            ("Query ages (months)", config.ages_query),
        ]
    )
    key_value_table("Build configuration", build_cfg)

    X_obs_z = (X_obs - X_obs_mean) / X_obs_std

    # Plot grid
    X_plot = np.linspace(X_obs.min(), X_obs.max(), config.n_plot).reshape(-1, 1)
    X_plot_z = (X_plot - X_obs_mean) / X_obs_std

    # Query grid
    X_query = np.array(config.ages_query).reshape(-1, 1)
    X_query_z = (X_query - X_obs_mean) / X_obs_std

    # Optional anchor point (Option D: per-draw zero of the GP at a reference age)
    anchor_g_u = bool(definition.anchor_g_u_at_ref)
    anchor_g_q = bool(definition.anchor_g_q_at_ref)
    use_gp_anchor = anchor_g_u or anchor_g_q
    if use_gp_anchor:
        if definition.gp_anchor_age_months is not None:
            anchor_age_months = float(definition.gp_anchor_age_months)
        else:
            anchor_age_months = (
                float(config.slope_anchors[0]) + float(config.slope_anchors[1])
            ) / 2.0
        X_anchor = np.array([[anchor_age_months]], dtype=float)
        X_anchor_z = (X_anchor - X_obs_mean) / X_obs_std
        X_all_z = np.vstack([X_obs_z, X_plot_z, X_query_z, X_anchor_z])
    else:
        anchor_age_months = None
        X_all_z = np.vstack([X_obs_z, X_plot_z, X_query_z])

    n_plot = X_plot_z.shape[0]
    n_query = X_query_z.shape[0]
    n_all = X_all_z.shape[0]
    i_anchor = (n + n_plot + n_query) if use_gp_anchor else None

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

    derived_rows: list[tuple[str, object]] = [
        ("HSGP basis size (m)", M),
        ("HSGP boundary factor (L)", L),
        ("Slope anchors (z-score)", (slope_age_a_z, slope_age_b_z)),
        ("Length-scale range (z-score)", (ell_low_z, ell_high_z)),
    ]
    if use_gp_anchor:
        derived_rows.append(
            (
                "GP anchor age (months)",
                f"{anchor_age_months:g} (g_u anchored: {anchor_g_u}, "
                f"g_q anchored: {anchor_g_q})",
            )
        )
    key_value_table("Derived quantities", derived_rows)

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
    if use_subject_codes:
        coords["subject_id"] = np.arange(n_subjects)

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

        if use_subject_codes:
            subject_obs = pm.Data(
                "subject_obs", subject_codes, dims=("obs_id",)
            )

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
        if anchor_g_u:
            g_unit_u_centred = g_unit_u - g_unit_u[i_anchor]
        else:
            g_unit_u_centred = g_unit_u
        g_u = pm.Deterministic("g_u", eta_u * g_unit_u_centred, dims=("all_id",))

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
        if anchor_g_q:
            g_unit_q_centred = g_unit_q - g_unit_q[i_anchor]
        else:
            g_unit_q_centred = g_unit_q
        g_q = pm.Deterministic("g_q", eta_q * g_unit_q_centred, dims=("all_id",))

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
        # Subject-level random intercepts (non-centered)
        # ============================================================

        if use_subject_re_u:
            tau_subj_u = pm.HalfNormal(
                "tau_subj_u", sigma=definition.tau_subj_u_sigma
            )
            delta_subj_u_raw = pm.Normal(
                "delta_subj_u_raw", mu=0.0, sigma=1.0, dims="subject_id"
            )
            delta_subj_u = pm.Deterministic(
                "delta_subj_u", tau_subj_u * delta_subj_u_raw, dims="subject_id"
            )
            subject_shift_u = delta_subj_u[subject_obs]
        else:
            subject_shift_u = 0.0

        if use_subject_re_q:
            tau_subj_q = pm.HalfNormal(
                "tau_subj_q", sigma=definition.tau_subj_q_sigma
            )
            delta_subj_q_raw = pm.Normal(
                "delta_subj_q_raw", mu=0.0, sigma=1.0, dims="subject_id"
            )
            delta_subj_q = pm.Deterministic(
                "delta_subj_q", tau_subj_q * delta_subj_q_raw, dims="subject_id"
            )
            subject_shift_q = delta_subj_q[subject_obs]
        else:
            subject_shift_q = 0.0

        # ============================================================
        # Observation-level quantities (with study effects)
        # ============================================================

        # Understood — obs level includes study shift (and optional subject shift)
        f_u_obs_re = f_u_all[i_obs0:i_obs1] + delta_u[study_obs] + subject_shift_u
        p_u_obs = pm.Deterministic(
            "p_u_obs", pm.math.sigmoid(f_u_obs_re), dims=("obs_id",)
        )
        # For diagnostics: population-level f at obs ages
        _ = pm.Deterministic("f_u_obs", f_u_all[i_obs0:i_obs1], dims=("obs_id",))

        # Production ratio — obs level includes study shift (and optional subject shift)
        h_obs_re = h_all[i_obs0:i_obs1] + delta_q[study_obs] + subject_shift_q
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

    render_model_graph(model_pm, context.reporting.output_dir)

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
    run_banner(definition.banner, subtitle=f"sampling config: {config}")

    env_info.report_environment_info()

    console.print()
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

    timings = context.timings
    run_started = time.perf_counter()

    with section("Prepare data", timings=timings):
        prepare_bivariate_re_data(context, definition)

    with section("Priors and hyperparameters", timings=timings):
        configure_bivariate_priors(context, definition)

    with section("Model definition and initialisation", timings=timings):
        build_model_re(context, definition)

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
