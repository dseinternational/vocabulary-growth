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

The study intercepts are implemented non-centred (delta = tau * z,
z ~ Normal(0, 1)) for HMC-friendly geometry; this is the same distribution as
above (see issue #65).

Plot and query predictions use the population-level trajectory (delta=0).
"""

import os
from collections.abc import Callable

import dse_research_utils.math.constants as math_constants
import dse_research_utils.statistics.descriptive as descriptive_stats
import dse_research_utils.statistics.models.data as model_data
import dse_research_utils.statistics.models.pymc_utils as pymc_utils
import numpy as np
import pymc as pm

import vocab_growth.data_utils as vocab_data_utils
from vocab_growth.models.build_utils import (
    construct_age_grids,
    slope_anchor_logit_coeffs,
    standardize_ages,
    validate_ell_bounds,
)
from vocab_growth.models.common import (
    get_hsgp_hyperparams,
    render_model_graph,
    report,
    run_fit_pipeline,
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
from vocab_growth.models.gp_utils import GPGrid, build_kappa_of_z, trend_and_gp
from vocab_growth.models.likelihood_utils import nested_outcome_spec
from vocab_growth.reporting import (
    dataframe_table,
    key_value_table,
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
    use_subject_codes = (
        definition.use_subject_re_u
        or definition.use_subject_re_q
        or definition.one_observation_per_subject
    )
    if use_subject_codes:
        columns = columns + ["subject_id"]
    if definition.exclude_us01_spoken_ceiling:
        columns = columns + ["survey_vocab_max"]

    df = vocab_data_utils.load_data(
        population=definition.population,
        columns=columns,
        sample_fraction=definition.sample_fraction,
        random_seed=definition.random_seed,
        max_age_months=definition.max_age_months,
    )
    ceiling_rows_excluded = 0
    if definition.exclude_us01_spoken_ceiling:
        df, ceiling_rows_excluded = (
            vocab_data_utils.exclude_us01_spoken_ceiling_rows(df)
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
    if use_subject_codes:
        vocab_data_utils.validate_subject_ids(analysis_df)
    n_before_single_administration = len(analysis_df)
    if definition.one_observation_per_subject:
        analysis_df = vocab_data_utils.select_one_observation_per_subject(
            analysis_df,
            random_seed=definition.random_seed,
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
    if definition.one_observation_per_subject:
        counts.append(
            (
                "Single-administration sensitivity",
                f"{n_before_single_administration} -> {len(analysis_df)} rows",
            )
        )
    if definition.exclude_us01_spoken_ceiling:
        counts.append(("us_01 WS-ceiling rows excluded", ceiling_rows_excluded))
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

_VALID_LAG_BASELINES = ("population", "within")


def _validate_cross_lag(lag_baseline: str, use_subject_re_u: bool) -> None:
    """Validate the VG16 within-child cross-lag configuration (issue #113).

    ``lag_baseline`` must be one of ``_VALID_LAG_BASELINES``. Both baselines are
    defined relative to the child's understood subject intercept — the
    within-child baseline subtracts it, the population-relative baseline adds it
    back — so ``use_subject_re_u`` must be True; otherwise the two baselines
    silently coincide (and the population branch would index a scalar). Raising
    here turns a silent misconfiguration into an explicit error.
    """
    if lag_baseline not in _VALID_LAG_BASELINES:
        raise ValueError(
            f"lag_baseline must be one of {_VALID_LAG_BASELINES}, got {lag_baseline!r}."
        )
    if not use_subject_re_u:
        raise ValueError(
            "Cross-lag (use_cross_lag=True) requires use_subject_re_u=True: both the "
            "population-relative and within-child baselines are defined relative to "
            "the child's understood subject intercept."
        )


def _compute_prev_wave_lag(analysis_df, n_trials: int):
    """Per-observation prior-wave understood lag source for the VG16 cross-lag.

    For each observation, locate the child's immediately-earlier age wave
    that carries an understood measure (the lag source). Returns
    ``(prev_idx, has_lag_f, y_u_prev_logit)`` as per-observation arrays:
    ``has_lag_f`` is 1.0 where such a prior wave exists and 0.0 otherwise (a
    child's first wave, or when every earlier wave lacks comprehension);
    ``prev_idx`` points at that prior wave (0 where absent, gated by
    ``has_lag_f``); ``y_u_prev_logit`` is the logit of the prior-wave understood
    proportion (clipped away from 0/1), and 0.0 where there is no lag source.
    """
    n = len(analysis_df)
    prev_idx = np.zeros(n, dtype=int)
    has_lag_f = np.zeros(n, dtype=float)
    subj = np.asarray(analysis_df["subject_code"], dtype=int)
    age = np.asarray(analysis_df["age"], dtype=float)
    und = analysis_df["understood"].to_numpy(dtype=float)
    row_order = np.arange(n)
    prev_subj, last, last_age = -1, -1, np.nan
    for pos in np.lexsort((row_order, age, subj)):  # walk each child in age order
        if subj[pos] != prev_subj:
            prev_subj, last, last_age = subj[pos], -1, np.nan
        if last >= 0 and age[pos] > last_age:
            prev_idx[pos] = last
            has_lag_f[pos] = 1.0
        if not np.isnan(und[pos]):
            last, last_age = pos, age[pos]
    und_prev = np.where(has_lag_f > 0, und[prev_idx], n_trials * 0.5)
    p_prev = np.clip(und_prev / n_trials, 1e-4, 1 - 1e-4)
    y_u_prev_logit = np.where(has_lag_f > 0, np.log(p_prev) - np.log(1 - p_prev), 0.0)
    return prev_idx, has_lag_f, y_u_prev_logit


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
    study_codes = np.asarray(analysis_df["study_code"], dtype=int)

    idx_u = np.where(has_u_train)[0]

    n = len(X_obs)
    n_u = len(y_u_observed)
    n_trials = context.model_data.n_trials
    spoken_spec = nested_outcome_spec(
        analysis_df,
        parent_col="understood",
        outcome_col="spoken",
        n_trials=n_trials,
        eligible_mask=~holdout,
    )
    if not np.array_equal(spoken_spec.indices, np.flatnonzero(has_s_train)):
        raise ValueError("Spoken likelihood rows do not match the training-data mask.")
    y_s_observed = spoken_spec.observed
    idx_s = spoken_spec.indices
    n_s = spoken_spec.n_observed
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

    # Cross-lag (VG16, issue #113): for each observation, the child's
    # immediately-earlier age wave with understood data is the lag source;
    # x_lag = 0 where there is no such prior wave (first wave, same-age
    # duplicates, or every earlier wave lacks comprehension).
    # prev_idx/has_lag_f/y_u_prev_logit are consumed below when injecting
    # beta_lag * x_lag into the q logit.
    use_cross_lag = bool(definition.use_cross_lag)
    prev_idx = np.zeros(n, dtype=int)
    has_lag_f = np.zeros(n, dtype=float)
    y_u_prev_logit = np.zeros(n, dtype=float)
    if use_cross_lag:
        _validate_cross_lag(definition.lag_baseline, use_subject_re_u)
        prev_idx, has_lag_f, y_u_prev_logit = _compute_prev_wave_lag(analysis_df, n_trials)
        print(
            f"Cross-lag ({definition.lag_baseline}): "
            f"{int(has_lag_f.sum())} of {n} observations have a prior-wave understood source."
        )

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
    X_obs_mean, X_obs_std, X_obs_z = standardize_ages(X_obs)

    build_cfg: list[tuple[str, object]] = [
        ("Total observations", n),
        ("Understood observed", n_u),
        ("Spoken observed", n_s),
        ("Spoken conditional on understood", spoken_spec.n_conditional),
        ("Spoken marginal fallback", spoken_spec.n_marginal),
        ("Spoken > understood violations", spoken_spec.n_parent_violations),
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

    # Plot / query grids (standardised), with the optional reference-age anchor
    # row — see models.build_utils.construct_age_grids.
    anchor_g_u = bool(definition.anchor_g_u_at_ref)
    anchor_g_q = bool(definition.anchor_g_q_at_ref)
    use_gp_anchor = anchor_g_u or anchor_g_q
    grids = construct_age_grids(
        X_obs,
        X_obs_z,
        X_obs_mean=X_obs_mean,
        X_obs_std=X_obs_std,
        n_plot=config.n_plot,
        ages_query=config.ages_query,
        slope_anchors=config.slope_anchors,
        use_gp_anchor=use_gp_anchor,
        gp_anchor_age_months=definition.gp_anchor_age_months,
        gp_domain_months=definition.gp_domain_months,
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

    L, M = get_hsgp_hyperparams(grids.X_gp_domain_z, ell_range_z)

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

        # Store masks and indices as constant data for extraction.
        # Use the *training* masks (full observed mask minus any holdout rows)
        # so the stored masks align with the likelihood rows / observed_data
        # consumed by extract_model_samples (issue #67). With no holdout column
        # has_*_train == has_*, so standard fits are unchanged.
        _ = pm.Data("obs_u_mask", has_u_train.astype(int), dims=("obs_id",))
        _ = pm.Data("obs_s_mask", has_s_train.astype(int), dims=("obs_id",))
        s_likelihood_n = pm.Data(
            "s_likelihood_n", spoken_spec.trials, dims=("obs_s_id",)
        )
        s_is_conditional = pm.Data(
            "s_is_conditional",
            spoken_spec.is_conditional.astype(int),
            dims=("obs_s_id",),
        )

        study_obs = pm.Data("study_obs", study_codes, dims=("obs_id",))

        if use_subject_codes:
            subject_obs = pm.Data(
                "subject_obs", subject_codes, dims=("obs_id",)
            )

        # Shared trend + HSGP builder (gp_utils); graph byte-identical to the
        # inlined form: stores g_u/f_u_all and g_q/h_all (+ slope/intercept/ell),
        # population-level latents (no study effect), Option-D anchor per outcome.
        gp_grid = GPGrid(
            sa_z=slope_age_a_z,
            sb_z=slope_age_b_z,
            ell_low_z=ell_low_z,
            ell_high_z=ell_high_z,
            M=M,
            L=L,
        )

        # ---- Understood (U) trajectory: f_U(a) -> p_U(a) ----
        f_u_all = trend_and_gp(
            cfg_low=config.p_slope_low_u_dist,
            cfg_hi=config.p_slope_hi_u_dist,
            cfg_ell=config.ell_unit_u_dist,
            cfg_eta=config.eta_u_dist,
            suffix="_u",
            X_all_z_data=X_all_z_data,
            grid=gp_grid,
            store_deterministic=True,
            latent_name="f_u_all",
            anchor_idx=i_anchor if anchor_g_u else None,
            n_obs=n,
        )

        # ---- Production ratio: h(a) -> q(a) = sigmoid(h(a)) ----
        h_all = trend_and_gp(
            cfg_low=config.p_slope_low_q_dist,
            cfg_hi=config.p_slope_hi_q_dist,
            cfg_ell=config.ell_unit_q_dist,
            cfg_eta=config.eta_q_dist,
            suffix="_q",
            X_all_z_data=X_all_z_data,
            grid=gp_grid,
            store_deterministic=True,
            latent_name="h_all",
            anchor_idx=i_anchor if anchor_g_q else None,
            n_obs=n,
        )

        # ============================================================
        # Study-level random intercepts
        # ============================================================

        # Non-centred, sum-to-zero (delta = tau * z, z ~ ZeroSumNormal) for
        # HMC-friendly geometry with few studies — consistent with the subject REs
        # below and the rest of the codebase. The tau * raw scaling keeps the
        # funnel-avoiding non-centring of issue #65; the sum-to-zero constraint on
        # the unit offsets additionally removes the intercept vs study-RE-mean ridge
        # (with few studies an unconstrained mean trades off against the global
        # intercept/slope, an R-hat failure). This is an intentional identifiability
        # constraint, not a prior-preserving reparameterisation — it removes the
        # group-mean DOF. We rescale sigma by sqrt(K/(K-1)) so each study effect's
        # marginal prior variance stays tau^2 (unchanged from independent Normal),
        # leaving only the mean DOF removed and a -1/K correlation imposed. Both
        # outcomes are informed by every retained study here, so a global zero-sum
        # over study_id is correct (cf. the joint model's per-outcome coordinates).
        # The public names delta_u/delta_q/tau_u/tau_q are preserved (downstream
        # scripts extract them by name from the trace).
        zsn_sigma = float(np.sqrt(n_studies / (n_studies - 1)))
        tau_u = pm.HalfNormal("tau_u", sigma=definition.tau_u_sigma)
        delta_u_raw = pm.ZeroSumNormal("delta_u_raw", sigma=zsn_sigma, dims="study_id")
        delta_u = pm.Deterministic("delta_u", tau_u * delta_u_raw, dims="study_id")

        tau_q = pm.HalfNormal("tau_q", sigma=definition.tau_q_sigma)
        delta_q_raw = pm.ZeroSumNormal("delta_q_raw", sigma=zsn_sigma, dims="study_id")
        delta_q = pm.Deterministic("delta_q", tau_q * delta_q_raw, dims="study_id")

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

        # Cross-lag (VG16, issue #113): the child's prior-wave understood
        # residual shifts their current production ratio q. The configured
        # baseline decides whether the residual is population-relative or
        # within-child. beta_lag > 0 means earlier receptive standing predicts
        # later expressive conversion; x_lag is 0 with no prior wave.
        if use_cross_lag:
            beta_lag = pm.Normal(
                "beta_lag", mu=definition.beta_lag_mu, sigma=definition.beta_lag_sigma
            )
            lag_base = f_u_obs_re[prev_idx]  # child's own expected understood logit at prior wave
            if definition.lag_baseline == "population":
                # Use the population+study baseline by removing the subject shift.
                lag_base = lag_base - subject_shift_u[prev_idx]
            x_lag = has_lag_f * (y_u_prev_logit - lag_base)
            q_lag_term = beta_lag * x_lag
        else:
            q_lag_term = 0.0

        # Production ratio — obs level includes study shift (and optional subject shift)
        h_obs_re = h_all[i_obs0:i_obs1] + delta_q[study_obs] + subject_shift_q + q_lag_term
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

        kappa_u_of_z = build_kappa_of_z(
            config.kappa_min_u_dist,
            config.a_kappa_u_dist,
            config.b_kappa_mag_u_dist,
            suffix="_u",
        )

        kappa_u_obs = pm.Deterministic(
            "kappa_u_obs", kappa_u_of_z(z_obs), dims="obs_id"
        )
        _ = pm.Deterministic("kappa_u_plot", kappa_u_of_z(z_plot), dims="plot_id")
        _ = pm.Deterministic("kappa_u_query", kappa_u_of_z(z_query), dims="query_id")

        # ============================================================
        # Kappa — spoken
        # ============================================================

        kappa_s_of_z = build_kappa_of_z(
            config.kappa_min_s_dist,
            config.a_kappa_s_dist,
            config.b_kappa_mag_s_dist,
            suffix="_s",
        )

        kappa_s_obs = pm.Deterministic(
            "kappa_s_obs", kappa_s_of_z(z_obs), dims="obs_id"
        )
        _ = pm.Deterministic("kappa_s_plot", kappa_s_of_z(z_plot), dims="plot_id")
        _ = pm.Deterministic("kappa_s_query", kappa_s_of_z(z_query), dims="query_id")

        # ============================================================
        # Likelihoods — separate observation indices
        # ============================================================
        #
        # Nested likelihood: where both outcomes are observed and S <= U,
        # spoken is modelled as S | U with U trials and mean q. Spoken-only rows
        # and source-data violations retain a marginal likelihood over the full
        # inventory with mean p_U * q.

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
        p_s_likelihood = pm.math.switch(
            s_is_conditional,
            q_obs[idx_s],
            p_s_obs[idx_s],
        )
        p_s_likelihood = pm.math.clip(p_s_likelihood, EPSILON, 1 - EPSILON)
        alpha_s = p_s_likelihood * kappa_s_obs[idx_s]
        beta_s = (1 - p_s_likelihood) * kappa_s_obs[idx_s]

        _ = pm.BetaBinomial(
            "y_s_obs",
            n=s_likelihood_n,
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


def bivariate_re_stages(
    definition: BivariateModelDefinition,
) -> list[tuple[str, Callable[[BivariateREContext], None]]]:
    """The ordered ``(stage name, stage fn)`` list for this engine's fit.

    Exposed separately from :func:`fit_bivariate_re_model` so a caller can
    substitute a single stage and still run the identical pipeline — the
    parameter-recovery harness swaps stage 0 (data preparation) for a loader
    that injects a simulated analysis frame (see
    :mod:`vocab_growth.recovery.refit`).
    """
    return [
        (
            "Prepare data",
            lambda ctx: prepare_bivariate_re_data(ctx, definition),
        ),
        (
            "Priors and hyperparameters",
            lambda ctx: configure_bivariate_priors(ctx, definition),
        ),
        (
            "Model definition and initialisation",
            lambda ctx: build_model_re(ctx, definition),
        ),
        ("Prior predictive checks", prior_predictive_checks),
        ("Posterior sampling", sample),
        ("Diagnostics", diagnostics),
        (
            "Posterior predictions",
            lambda ctx: sample_posterior_predictive(ctx, definition),
        ),
        ("Posterior summary", posterior_summary),
        (
            "Plots",
            lambda ctx: _run_bivariate_joint_plots(ctx, definition),
        ),
        ("Report", report),
    ]


def fit_bivariate_re_model(
    config: str,
    definition: BivariateModelDefinition,
) -> BivariateREContext:
    """
    Fit pipeline for bivariate model with study random intercepts (VG07).
    """
    return run_fit_pipeline(config, definition, stages=bivariate_re_stages(definition))
