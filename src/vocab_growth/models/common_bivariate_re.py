# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Bivariate vocabulary growth model with study-level random intercepts.

This is the engine for eleven of the twenty registered models — VG07-VG10,
VG13, VG16 and VG19-VG23 — and the catalogue
(:mod:`vocab_growth.models.catalogue`) is the authoritative mapping. It extends
the production-ratio reparameterization from common_bivariate with study-level
random intercepts on both the understood trajectory and the production ratio:

    f_U(a, s) = mean_trend_u(a) + g_u(a) + delta_u[s]
    h(a, s)   = mean_trend_q(a) + g_q(a) + delta_q[s]

    delta_u[s] = tau_u * z_u[s],  z_u ~ ZeroSumNormal(sqrt(K / (K - 1)))
    delta_q[s] = tau_q * z_q[s],  z_q ~ ZeroSumNormal(sqrt(K / (K - 1)))

over the K retained studies. The ``tau * z`` scaling is the funnel-avoiding
non-centring of issue #65; the sum-to-zero constraint on the unit offsets is a
deliberate **identifiability** constraint on top of it, not a prior-preserving
reparameterisation — it removes the group-mean degree of freedom that otherwise
trades off against the global intercept. The ``sqrt(K / (K - 1))`` rescaling
keeps each study effect's marginal prior variance at ``tau^2``, so the marginals
match an independent ``Normal(0, tau)`` while the joint does not: a ``-1/(K-1)``
correlation is imposed. See the full argument at the construction site in
:func:`build_model_re`.

Plot and query predictions use the population-level trajectory (delta=0).
"""

import os
from collections.abc import Callable

import dse_research_utils.math.constants as math_constants
import dse_research_utils.statistics.descriptive as descriptive_stats
import dse_research_utils.statistics.models.data as model_data
import dse_research_utils.statistics.models.pymc_utils as pymc_utils
import dse_research_utils.statistics.models.reporting as reporting
import dse_research_utils.statistics.models.sampling as sampling
import numpy as np
import pandas as pd
import pymc as pm

import vocab_growth.data_utils as vocab_data_utils
from vocab_growth.models import subject_effects
from vocab_growth.models.build_utils import (
    construct_age_grids,
    require_valid_counts,
    standardize_ages,
    standardize_anchor_ages,
    validate_ell_bounds,
)
from vocab_growth.models.common import (
    ModelFitContext,
    build_kappa_for_config,
    get_hsgp_hyperparams,
    kappa_anchor_derived_rows,
    render_model_graph,
    report,
    run_fit_pipeline,
)
from vocab_growth.models.common_bivariate import (
    BivariateContext,
    configure_bivariate_priors,
    diagnostics,
    posterior_summary,
    prior_predictive_checks,
    run_bivariate_joint_plots,
    sample,
    sample_posterior_predictive,
)
from vocab_growth.models.cross_lag import (
    cross_lag_audit_frame,
    prev_wave_lag_for_frame,
    report_cross_lag_support,
    validate_cross_lag,
)
from vocab_growth.models.definitions import (
    BivariateModelDefinition,
    clamp_targets,
)
from vocab_growth.models.gp_utils import (
    GPGrid,
    build_child_factor,
    build_child_slope,
    build_subject_scale_of_z,
    trend_and_gp,
)
from vocab_growth.models.likelihood_utils import nested_outcome_alpha_beta
from vocab_growth.models.observation_arrays import prepare_bivariate_observations
from vocab_growth.reporting import (
    dataframe_table,
    key_value_table,
)

EPSILON = math_constants.EPSILON

BivariateREContext = BivariateContext


# ============================================================
# Data preparation (with study column)
# ============================================================


def build_bivariate_re_analysis_frame(
    definition: BivariateModelDefinition,
) -> tuple[pd.DataFrame, dict]:
    """The exact prepared frame the bivariate-RE engine fits, no side effects.

    Split out of :func:`prepare_bivariate_re_data` so fitted-output validation
    can recompute the frame (and its exact hash) without a fit context — see
    :mod:`vocab_growth.analysis_frames` (issue #266 finding 1).
    """
    columns = ["age", "understood", "spoken", "study"]
    use_subject_codes = (
        definition.use_subject_re_u
        or definition.use_subject_re_q
        or definition.one_observation_per_subject
    )
    if use_subject_codes:
        columns = columns + ["subject_id"]
    # `survey_vocab_max` is kept for the whole Down syndrome pool, not just the
    # defect rules and the cross-lag audit that used to be its only callers: the
    # observed-trajectory overlay on the median-trend plots cannot tell a real
    # reversal from a change of recording form without it. Safe there because
    # `load_data` returns `df[columns]` for this population -- a pure projection
    # -- the column has no missing values, and the only row filter here is
    # `dropna(subset=["age"])`, so asking for it changes no fit.
    #
    # **Down syndrome only.** The typically-developing frame is built from a
    # Wordbank query that never produces this column, so requesting it there
    # raises `KeyError`. The other three flags are DS-only in practice; the
    # population test is what actually makes this safe for VG11-VG13 and VG21.
    if (
        definition.exclude_us01_spoken_ceiling
        or definition.dse_native_only
        # The cross-lag audit records checklist-form transitions between the
        # source and current waves (issue #242); the ceiling column is the
        # form identity the frame carries. Loading it changes no likelihood
        # input and no raw-data fingerprint.
        or definition.use_cross_lag
        or definition.population is vocab_data_utils.Population.DOWN_SYNDROME
    ):
        columns = columns + ["survey_vocab_max"]

    df = vocab_data_utils.load_data(
        population=definition.population,
        columns=columns,
        sample_fraction=definition.sample_fraction,
        random_seed=definition.random_seed,
        # TD language scope is part of the model graph; DS ignores it.
        languages=getattr(
            definition, "td_languages", vocab_data_utils.ENGLISH_LANGUAGES
        ),
        max_age_months=definition.max_age_months,
        include_implausible_production=definition.include_implausible_production,
    )
    ceiling_rows_excluded = 0
    if definition.exclude_us01_spoken_ceiling:
        df, ceiling_rows_excluded = (
            vocab_data_utils.exclude_us01_spoken_ceiling_rows(df)
        )
    non_native_rows_excluded = 0
    if definition.dse_native_only:
        df, non_native_rows_excluded = (
            vocab_data_utils.restrict_to_dse_native_administrations(df)
        )
    excluded_study_rows = 0
    if definition.exclude_studies:
        keep = ~df["study"].isin(definition.exclude_studies)
        excluded_study_rows = int((~keep).sum())
        if excluded_study_rows == 0:
            raise ValueError(
                f"exclude_studies={definition.exclude_studies!r} matched no rows. "
                "A leave-one-study-out check that removes nothing cannot fail, "
                "which is worse than not running it -- check the study codes."
            )
        df = df[keep]
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

    return analysis_df, {
        "use_subject_codes": use_subject_codes,
        "ceiling_rows_excluded": ceiling_rows_excluded,
        "non_native_rows_excluded": non_native_rows_excluded,
        "excluded_study_rows": excluded_study_rows,
        "dropped_studies": dropped_studies,
        "n_before_single_administration": n_before_single_administration,
        "unique_studies": unique_studies,
        "n_subjects": n_subjects,
    }


def prepare_bivariate_re_data(
    context: BivariateREContext,
    definition: BivariateModelDefinition,
):
    """Load and prepare data for a bivariate model with study random effects."""
    analysis_df, info = build_bivariate_re_analysis_frame(definition)
    ceiling_rows_excluded = info["ceiling_rows_excluded"]
    non_native_rows_excluded = info["non_native_rows_excluded"]
    excluded_study_rows = info["excluded_study_rows"]
    dropped_studies = info["dropped_studies"]
    n_before_single_administration = info["n_before_single_administration"]
    unique_studies = info["unique_studies"]
    n_subjects = info["n_subjects"]

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
    if definition.exclude_studies:
        counts.append(
            (
                f"Studies excluded ({', '.join(definition.exclude_studies)})",
                f"{excluded_study_rows} rows",
            )
        )
    if definition.exclude_us01_spoken_ceiling:
        counts.append(("us_01 WS-ceiling rows excluded", ceiling_rows_excluded))
    if definition.dse_native_only:
        # Logged because a zero here means the variant has stopped biting and is
        # silently fitting the model of record's data, which is a failure that
        # looks exactly like a pass.
        counts.append(("Non-native-ceiling rows excluded", non_native_rows_excluded))
    if definition.include_implausible_production:
        counts.append((
            "us_01 implausible production reinstated",
            vocab_data_utils.count_reinstated_implausible_production(
                definition.max_age_months
            ),
        ))
    key_value_table("Observation counts", counts)
    dataframe_table(desc, title="Descriptive statistics")

    # Create a BinomialModelData for the context interface
    X_obs = np.asarray(analysis_df["age"], dtype=float).reshape(-1, 1)
    y_u_valid = analysis_df.loc[analysis_df["understood"].notna(), "understood"]
    require_valid_counts(
        y_u_valid.to_numpy(dtype=float), "understood", definition.n_trials
    )
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

    # Which of the five child-effect structures this definition selects, and
    # every rejection that goes with them, resolved before the model context is
    # entered so a refusal fires against a definition rather than part-way
    # through a half-built graph (issue #273).
    plan = subject_effects.resolve(definition)
    use_subject_re_u = plan["u"].is_active
    use_subject_re_q = plan["q"].is_active
    use_subject_codes = plan.any_active

    # Everything the likelihood is assembled from, derived from the frame in one
    # pure step (`observation_arrays`). Separated for the reasons its module
    # docstring gives: the spoken likelihood mask, the pre-cast count validation
    # and the held-out mask each have a specific past failure behind them, and
    # none could be tested without building a model.
    n_trials = context.model_data.n_trials
    observations = prepare_bivariate_observations(
        analysis_df,
        definition,
        n_trials=n_trials,
        use_subject_codes=use_subject_codes,
    )
    X_obs = observations.X_obs
    y_u_observed = observations.y_u_observed
    idx_u = observations.idx_u
    y_s_observed = observations.y_s_observed
    idx_s = observations.idx_s
    has_u_train = observations.has_u_likelihood
    has_s_likelihood = observations.has_s_likelihood
    spoken_spec = observations.spoken_spec
    spoken_fallback = observations.spoken_fallback
    n_fallback_dropped = observations.n_fallback_dropped
    study_codes = observations.study_codes
    subject_codes = observations.subject_codes
    n_subjects = observations.n_subjects
    n = observations.n
    n_u = observations.n_u
    n_s = observations.n_s
    n_studies = observations.n_studies

    # Cross-lag (VG16, issue #113): the child's most recent strictly earlier
    # administration wave with understood data is the lag source, computed
    # over complete (subject, age) wave groups (issue #242); x_lag = 0 where
    # there is no such prior wave (a child's first wave, or every earlier wave
    # lacks comprehension). prev_idx/has_lag_f/y_u_prev_logit are consumed
    # below when injecting beta_lag * x_lag into the q logit.
    use_cross_lag = bool(definition.use_cross_lag)
    prev_idx = np.zeros(n, dtype=int)
    has_lag_f = np.zeros(n, dtype=float)
    y_u_prev_logit = np.zeros(n, dtype=float)
    if use_cross_lag:
        validate_cross_lag(definition.lag_baseline, use_subject_re_u)
        prev_idx, has_lag_f, y_u_prev_logit = prev_wave_lag_for_frame(
            analysis_df, n_trials, definition
        )
        print(
            f"Cross-lag ({definition.lag_baseline}): "
            f"{int(has_lag_f.sum())} of {n} observations have a prior-wave understood source."
        )
        report_cross_lag_support(
            context.reporting.output_dir,
            cross_lag_audit_frame(
                analysis_df,
                prev_idx,
                has_lag_f,
                spoken_spec.indices,
                spoken_spec.is_conditional,
            ),
            n_obs=n,
        )

    # Range validation happens ONCE, before the integer cast, and not here: the cast
    # truncates silently, so a post-cast bound cannot catch 810.9 or -0.1, which
    # truncate into range. `build_utils.require_valid_counts` covers the parent
    # column and `likelihood_utils.nested_outcome_spec` covers each nested one, both
    # on the pre-cast floats (#236, #240).

    # Standardise ages
    X_obs_mean, X_obs_std, X_obs_z = standardize_ages(X_obs)

    build_cfg: list[tuple[str, object]] = [
        ("Total observations", n),
        ("Understood observed", n_u),
        ("Spoken observed", n_s),
        ("Spoken conditional on understood", spoken_spec.n_conditional),
        ("Spoken marginal fallback", spoken_spec.n_marginal),
        ("Spoken fallback treatment", spoken_fallback),
        ("Spoken fallback rows dropped", n_fallback_dropped),
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
    slope_age_a_z, slope_age_b_z = standardize_anchor_ages(
        config.slope_anchors, X_obs_mean=X_obs_mean, X_obs_std=X_obs_std
    )

    derived_rows: list[tuple[str, object]] = [
        ("HSGP basis size (m)", M),
        ("HSGP boundary factor (L)", L),
        ("Slope anchors (z-score)", (slope_age_a_z, slope_age_b_z)),
        ("Length-scale range (z-score)", (ell_low_z, ell_high_z)),
        *kappa_anchor_derived_rows(
            config, X_obs_mean=X_obs_mean, X_obs_std=X_obs_std, suffix="_u"
        ),
        *kappa_anchor_derived_rows(
            config, X_obs_mean=X_obs_mean, X_obs_std=X_obs_std, suffix="_s"
        ),
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
    # VG19: the two per-child effects (offset at the reference age, and rate).
    # Declared unconditionally -- an unused coord adds no variable to the graph,
    # and a conditional would have to re-derive what the plan already knows.
    coords["child_effect"] = np.array(["intercept", "slope"])
    # VG22: the four effects the low-rank factor spans, and its latent
    # dimensions. `child_effect4_b` is the second axis of the 4x4 correlation --
    # ArviZ needs two distinct dim names for a square matrix.
    coords["child_effect4"] = np.array(
        ["u_intercept", "u_slope", "q_intercept", "q_slope"]
    )
    coords["child_effect4_b"] = np.array(
        ["u_intercept", "u_slope", "q_intercept", "q_slope"]
    )
    coords["factor"] = np.arange(plan.factor.rank if plan.factor else 1)

    with pm.Model(coords=coords) as model_pm:

        # ---- Data ----

        X_all_z_data = pm.Data("X_all_z", X_all_z, dims=("all_id", "x_dim"))

        _ = pm.Data("X_obs", X_obs.flatten(), dims=("obs_id",))
        _ = pm.Data("X_plot", X_plot.flatten(), dims=("plot_id",))
        _ = pm.Data("X_query", X_query.flatten(), dims=("query_id",))

        # Store masks and indices as constant data for extraction.
        # Use the *likelihood* masks (full observed mask minus any holdout
        # rows, minus any rows the fallback treatment removed from the spoken
        # likelihood) so the stored masks align with the likelihood rows /
        # observed_data consumed by extract_model_samples (issues #67, #266).
        # With no holdout column and a non-dropping fallback treatment these
        # equal has_u / has_s, so standard fits are unchanged.
        _ = pm.Data("obs_u_mask", has_u_train.astype(int), dims=("obs_id",))
        _ = pm.Data("obs_s_mask", has_s_likelihood.astype(int), dims=("obs_id",))
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

        # One flag, two means: see definitions.clamp_targets. 'q_only' is
        # truthy, so testing the raw value would clamp both.
        _clamp_u, _clamp_q = clamp_targets(
            definition.clamp_mean_above_hi_anchor
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
            clamp_above_hi=_clamp_u,
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
            clamp_above_hi=_clamp_q,
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
        # leaving only the mean DOF removed and a -1/(K-1) correlation imposed. Both
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

        # Which of the five child-effect structures this definition selects, and
        # every rejection that goes with them, was resolved by
        # `subject_effects.resolve` before this context was entered (issue
        # #273). What is left here is the graph each resolved kind emits.
        #
        # Proposal A1 (registered sensitivity): where a subject-scale field
        # carries an `AgeVaryingSubjectScale` instead of a scalar, the per-child
        # deviate is scaled by tau(age) at each observation's own age and the
        # paired kappa block is held flat. The scalar path below is untouched and
        # emits exactly the ops it always did, so every model of record keeps its
        # graph. `tau_*_of_z` is carried forward to emit the plot/query scales
        # once the standardised grids exist.
        spec_u = plan["u"].age_varying
        spec_q = plan["q"].age_varying
        # VG19: the same overloaded field can instead carry a child slope, which
        # is a different age function through the seam A1 opened.
        slope_u = plan["u"].slope
        slope_q = plan["q"].slope
        slope_ref_age = plan.slope_ref_age_months
        corr_eta = plan.correlation_eta
        # VG22: a low-rank factor over all four child effects. Built once, ahead
        # of the per-outcome branches, because unlike every other subject
        # structure here it spans both outcomes -- the whole point of the form is
        # that one child's comprehension standing and production-ratio rate are
        # driven by shared latent dimensions. The per-outcome branches below then
        # consume the shifts it returns rather than building their own.
        if plan.factor is not None:
            # The two reference-age scales it also returns are deliberately
            # discarded: for a factor the between-child geometry lives in the
            # loading matrix, so there is no scalar scale for a downstream term to
            # use. `build_child_factor` stores them as named Deterministics, which
            # is how the summaries reach them.
            factor_shift_u, factor_shift_q, _, _ = (
                build_child_factor(
                    plan.factor,
                    tau0_u_sigma=definition.tau_subj_u_sigma,
                    tau0_q_sigma=definition.tau_subj_q_sigma,
                    age_obs_months=X_obs.flatten(),
                    subject_obs=subject_obs,
                )
            )
        else:
            factor_shift_u = factor_shift_q = None

        tau_u_of_z = tau_q_of_z = None
        # Built here rather than reusing the named `z_obs` Deterministic, which is
        # created further down: reordering that would change every model's graph.
        z_obs_raw = (
            X_all_z_data[i_obs0:i_obs1, 0]
            if (spec_u is not None or spec_q is not None)
            else None
        )

        if use_subject_re_u:
            # Only the plain-HalfNormal branch below yields a scalar between-child
            # scale, and only it needs one -- for its own `delta_subj_u`. The factor
            # branch's geometry is a loading matrix, the slope branch's is a
            # (tau0, tau1, rho) triple, and the A1 branch's is a function of age.
            # None of those is bound to a shared name here, so the shape of this
            # block cannot suggest that a later term may reach for "the" scale.
            if factor_shift_u is not None:
                subject_shift_u = factor_shift_u
            elif slope_u is not None:
                subject_shift_u, _ = build_child_slope(
                    slope_u,
                    age_obs_months=X_obs.flatten(),
                    subject_obs=subject_obs,
                    ref_age_months=slope_ref_age,
                    name="tau_subj_u",
                )
            elif spec_u is None:
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
                tau_u_of_z, tau_subj_u_young = build_subject_scale_of_z(
                    spec_u,
                    anchor_z=standardize_anchor_ages(
                        spec_u.anchor_ages,
                        X_obs_mean=X_obs_mean,
                        X_obs_std=X_obs_std,
                    ),
                    name="tau_subj_u",
                )
                delta_subj_u_raw = pm.Normal(
                    "delta_subj_u_raw", mu=0.0, sigma=1.0, dims="subject_id"
                )
                # `delta_subj_u` keeps its name and its per-child meaning, read at
                # the young anchor; the shift applied to the likelihood is the
                # age-scaled one.
                _ = pm.Deterministic(
                    "delta_subj_u",
                    tau_subj_u_young * delta_subj_u_raw,
                    dims="subject_id",
                )
                subject_shift_u = tau_u_of_z(z_obs_raw) * delta_subj_u_raw[subject_obs]
        else:
            subject_shift_u = 0.0

        if use_subject_re_q:
            # As for `u` above: only the plain-HalfNormal branch produces a
            # scalar scale, and only it consumes one.
            if factor_shift_q is not None:
                subject_shift_q = factor_shift_q
            elif slope_q is not None:
                # `corr_eta` is guaranteed None here: the resolver refuses the
                # combination, so no branch on it is needed or wanted.
                subject_shift_q, _ = build_child_slope(
                    slope_q,
                    age_obs_months=X_obs.flatten(),
                    subject_obs=subject_obs,
                    ref_age_months=slope_ref_age,
                    name="tau_subj_q",
                )
            elif spec_q is None:
                tau_subj_q = pm.HalfNormal(
                    "tau_subj_q", sigma=definition.tau_subj_q_sigma
                )
                delta_subj_q_raw = pm.Normal(
                    "delta_subj_q_raw", mu=0.0, sigma=1.0, dims="subject_id"
                )
                if corr_eta is None:
                    delta_subj_q_value = tau_subj_q * delta_subj_q_raw
                else:
                    # VG20 (issue #224). A child's two deviations are drawn from a
                    # joint Normal rather than independently, in Cholesky form so
                    # the nesting is exact: at rho_uq = 0 this is the expression
                    # above, op for op.
                    #
                    # `delta_subj_u_raw` is reused as the shared first coordinate
                    # and `delta_subj_q_raw` becomes the whitened second one, so
                    # both keep their names, their standard-Normal priors and
                    # their dims. Every downstream reader of `delta_subj_u` and
                    # `delta_subj_q` — the summaries, the comparison suite, the
                    # recovery scorer — sees what it always saw.
                    rho_raw = pm.Beta(
                        "rho_uq_raw", alpha=corr_eta, beta=corr_eta
                    )
                    rho_uq = pm.Deterministic("rho_uq", 2.0 * rho_raw - 1.0)
                    delta_subj_q_value = tau_subj_q * (
                        rho_uq * delta_subj_u_raw
                        + pm.math.sqrt(1.0 - rho_uq**2) * delta_subj_q_raw
                    )
                delta_subj_q = pm.Deterministic(
                    "delta_subj_q", delta_subj_q_value, dims="subject_id"
                )
                subject_shift_q = delta_subj_q[subject_obs]
            else:
                tau_q_of_z, tau_subj_q_young = build_subject_scale_of_z(
                    spec_q,
                    anchor_z=standardize_anchor_ages(
                        spec_q.anchor_ages,
                        X_obs_mean=X_obs_mean,
                        X_obs_std=X_obs_std,
                    ),
                    name="tau_subj_q",
                )
                delta_subj_q_raw = pm.Normal(
                    "delta_subj_q_raw", mu=0.0, sigma=1.0, dims="subject_id"
                )
                _ = pm.Deterministic(
                    "delta_subj_q",
                    tau_subj_q_young * delta_subj_q_raw,
                    dims="subject_id",
                )
                subject_shift_q = tau_q_of_z(z_obs_raw) * delta_subj_q_raw[subject_obs]
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

        # Proposal A1's age-varying subject scale, reported on the same grids as
        # kappa so the two can be read against each other — which is the whole
        # point of the variant.
        if tau_u_of_z is not None:
            _ = pm.Deterministic(
                "tau_subj_u_plot", tau_u_of_z(z_plot), dims="plot_id"
            )
            _ = pm.Deterministic(
                "tau_subj_u_query", tau_u_of_z(z_query), dims="query_id"
            )
        if tau_q_of_z is not None:
            _ = pm.Deterministic(
                "tau_subj_q_plot", tau_q_of_z(z_plot), dims="plot_id"
            )
            _ = pm.Deterministic(
                "tau_subj_q_query", tau_q_of_z(z_query), dims="query_id"
            )

        # ============================================================
        # Kappa — understood
        # ============================================================

        kappa_u_of_z = build_kappa_for_config(
            config,
            X_obs_mean=X_obs_mean,
            X_obs_std=X_obs_std,
            suffix="_u",
            hold_constant=spec_u is not None and spec_u.hold_kappa_constant,
        )

        kappa_u_obs = pm.Deterministic(
            "kappa_u_obs", kappa_u_of_z(z_obs), dims="obs_id"
        )
        _ = pm.Deterministic("kappa_u_plot", kappa_u_of_z(z_plot), dims="plot_id")
        _ = pm.Deterministic("kappa_u_query", kappa_u_of_z(z_query), dims="query_id")

        # ============================================================
        # Kappa — spoken
        # ============================================================

        kappa_s_of_z = build_kappa_for_config(
            config,
            X_obs_mean=X_obs_mean,
            X_obs_std=X_obs_std,
            suffix="_s",
            hold_constant=spec_q is not None and spec_q.hold_kappa_constant,
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

        # Spoken likelihood (only where observed). Both bivariate engines route
        # through the one helper so their graphs cannot drift apart.
        alpha_s, beta_s = nested_outcome_alpha_beta(
            treatment=spoken_fallback,
            is_conditional=s_is_conditional,
            conditional_p=q_obs[idx_s],
            marginal_p=p_s_obs[idx_s],
            parent_p=p_u_obs[idx_s],
            parent_kappa=kappa_u_obs[idx_s],
            kappa=kappa_s_obs[idx_s],
            epsilon=EPSILON,
            outcome="s",
            fallback_kappa_sigma=definition.spoken_fallback_kappa_sigma,
        )

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
        (
            "Prior predictive checks",
            # Passes the definition so the child-level checks can dispatch on
            # the child-effect structure (#233); the population figures do not
            # need it.
            lambda ctx: prior_predictive_checks(ctx, definition),
        ),
        ("Posterior sampling", sample),
        # The definition travels with the stage so the cross-lag models can
        # reorder the pair plot and suppress the leaking understood LOO
        # (issue #242); every other model takes the default path unchanged.
        ("Diagnostics", lambda ctx: diagnostics(ctx, definition)),
        (
            "Posterior predictions",
            lambda ctx: sample_posterior_predictive(ctx, definition),
        ),
        ("Posterior summary", posterior_summary),
        (
            "Plots",
            lambda ctx: run_bivariate_joint_plots(ctx, definition),
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


def rebuild_model_context(
    definition: BivariateModelDefinition,
    *,
    output_dir: str,
    sampling_config: str = "dev",
) -> BivariateREContext:
    """Build ``definition``'s model graph without sampling it.

    For readers of a stored fit that must recompute a posterior quantity the
    trace does not carry -- since 2026-08-23 the observation-sized
    deterministics are not sampled (``fit_artifacts.sampled_variable_names``),
    and ``vocab_growth.posterior_recompute.with_deterministics`` rebuilds them
    from the stored free parameters given the graph. It runs the engine's own
    data preparation and build, so the observation order and every data rule
    are exactly those of a fit of ``definition``; the caller still has to check
    that the fit *was* of this definition on the current data
    (``fit_artifacts.validate_fit_output``) before aligning anything by row.

    ``output_dir`` receives the preparation stage's descriptive-statistics CSV
    and should be scratch. ``sampling_config`` only fills the context's
    sampling configuration, which the build does not read.
    """
    context: BivariateREContext = ModelFitContext(
        reporting=reporting.ReportingConfiguration(
            model_name=definition.model_id,
            config_name=definition.config_name,
            output_root_dir=output_dir,
            ci_prob=0.89,
            interval_kind="eti",
        ),
        sampling=sampling.get_sampling_configuration(sampling_config),
        sampling_config_name=sampling_config,
    )
    os.makedirs(context.reporting.output_dir, exist_ok=True)
    prepare_bivariate_re_data(context, definition)
    configure_bivariate_priors(context, definition)
    build_model_re(context, definition)
    return context
