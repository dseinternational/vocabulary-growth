# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Univariate vocabulary growth model with study-level random intercepts.

Extends the single-outcome HSGP model from common.py with dataset-level
random intercepts on the outcome trajectory:

    f(a, s) = mean_trend(a) + g(a) + delta[s]

    delta[s] = tau * delta_raw[s],  delta_raw ~ ZeroSumNormal(sqrt(K / (K - 1)))
    tau ~ HalfNormal(tau_study_sigma)

over the K retained studies. The sum-to-zero constraint is a deliberate
**identifiability** constraint, not a prior-preserving reparameterisation: it
removes the group-mean degree of freedom that otherwise trades off against the
global intercept. The ``sqrt(K / (K - 1))`` rescaling keeps each study effect's
marginal prior variance at ``tau^2``, so the marginals match an independent
``Normal(0, tau)`` while the joint does not — the K effects carry a ``-1/(K-1)``
correlation.
``centred_study_re`` selects an equivalent centred coordinate for the same
distribution. See the full argument at the construction site in
:func:`build_univariate_re_model`.

Plot and query predictions use the population-level trajectory (delta = 0).

This module is the shared pipeline for VG11 (TD spoken) and VG12 (TD understood).
"""

import os
import sys
from collections.abc import Callable

import dse_research_utils.math.constants as math_constants
import dse_research_utils.statistics.descriptive as descriptive_stats
import dse_research_utils.statistics.models.data as model_data
import dse_research_utils.statistics.models.pymc_utils as pymc_utils
import numpy as np
import pandas as pd
import pymc as pm

import vocab_growth.data_utils as vocab_data_utils
import vocab_growth.reporting_ages as reporting_ages
from vocab_growth.fit_artifacts import save_trace
from vocab_growth.models import sex_covariate, subject_effects
from vocab_growth.models.build_reporting import BuildReport
from vocab_growth.models.build_utils import (
    construct_age_grids,
    require_valid_counts,
    standardize_ages,
    standardize_anchor_ages,
    validate_ell_bounds,
)
from vocab_growth.models.calibration import write_trace_calibration
from vocab_growth.models.common import (
    ModelFitContext,
    build_kappa_for_config,
    configure_univariate_priors,
    diagnostics,
    extract_model_samples,
    get_hsgp_hyperparams,
    kappa_anchor_derived_rows,
    posterior_summary,
    prior_predictive_checks,
    report,
    report_model_build,
    run_fit_pipeline,
    run_standard_plots,
    sample,
)
from vocab_growth.models.common import (
    sample_posterior_predictive as _base_sample_posterior_predictive,
)
from vocab_growth.models.definitions import UnivariateModelDefinition
from vocab_growth.models.gp_utils import (
    GPGrid,
    build_subject_scale_of_z,
    build_variance_partition,
    trend_and_gp,
)
from vocab_growth.models.observation_arrays import sex_contrast_codes
from vocab_growth.models.study_effects import zero_sum_study_offsets
from vocab_growth.models.subject_effects import UNIVARIATE_OUTCOME
from vocab_growth.models.subject_marginal import (
    partition_subject_rows,
    singleton_first_order,
    subject_marginal_betabinomial,
    zero_padded_subject_shift,
)
from vocab_growth.posterior_analysis import (
    extract_posterior,
    extract_posterior_predictive,
    extract_posterior_predictive_float,
)
from vocab_growth.reporting import (
    dataframe_table,
    key_value_table,
)

EPSILON = math_constants.EPSILON

# Type alias — the RE variant uses the same context type as the base model.
UnivariateREContext = ModelFitContext


# ============================================================
# Data preparation (with study column)
# ============================================================


def build_univariate_re_analysis_frame(
    definition: UnivariateModelDefinition,
) -> tuple[pd.DataFrame, dict]:
    """The exact prepared frame the univariate-RE engine fits, no side effects.

    Split out of :func:`prepare_univariate_re_data` so fitted-output validation
    can recompute the frame (and its exact hash) without a fit context — see
    :mod:`vocab_growth.analysis_frames` (issue #266 finding 1).
    """
    y_col = definition.outcome.value
    use_subject_codes = (
        definition.use_subject_re or definition.one_observation_per_subject
    )
    columns = ["age", y_col, "study"]
    if use_subject_codes:
        columns.append("subject_id")
    # Sex (#324): carried only when the definition asks for a sex term, so every
    # other univariate frame is untouched. Rows of unrecorded sex are kept.
    if sex_covariate.sex_effect_sigma(definition) is not None:
        columns.append("sex")

    df = vocab_data_utils.load_data(
        population=definition.population,
        columns=columns,
        sample_fraction=definition.sample_fraction,
        random_seed=definition.random_seed,
        # TD language scope is part of the model graph; DS ignores it.
        languages=getattr(
            definition, "td_languages", vocab_data_utils.ENGLISH_LANGUAGES
        ),
    )
    analysis_df = df[columns].dropna(subset=["age", y_col]).reset_index(drop=True)
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

    # Integer study codes (sorted for reproducibility)
    unique_studies = sorted(analysis_df["study"].unique())
    study_map = {s: i for i, s in enumerate(unique_studies)}
    analysis_df["study_code"] = analysis_df["study"].map(study_map).astype(int)

    n_subjects: int | None = None
    n_repeated_subjects: int | None = None
    if use_subject_codes:
        subject_keys = (
            analysis_df["study"].astype(str)
            + "::"
            + analysis_df["subject_id"].astype(str)
        )
        analysis_df["subject_key"] = subject_keys
        unique_subjects = sorted(subject_keys.unique())
        subject_map = {subject: i for i, subject in enumerate(unique_subjects)}
        analysis_df["subject_code"] = subject_keys.map(subject_map).astype(int)
        n_subjects = len(unique_subjects)
        n_repeated_subjects = int(
            (analysis_df.groupby("subject_code").size() > 1).sum()
        )
        if getattr(definition, "singleton_marginalisation", None) is not None:
            # The marginalised likelihood reads its two blocks as slices, so the
            # rows whose child effect it integrates out come first. Stable, and
            # applied here so every downstream array -- ages, counts, the
            # calibration table -- carries the same order.
            analysis_df = (
                analysis_df.iloc[
                    singleton_first_order(analysis_df["subject_code"].to_numpy())
                ]
                .reset_index(drop=True)
            )

    return analysis_df, {
        "use_subject_codes": use_subject_codes,
        "dropped_studies": dropped_studies,
        "n_before_single_administration": n_before_single_administration,
        "unique_studies": unique_studies,
        "n_subjects": n_subjects,
        "n_repeated_subjects": n_repeated_subjects,
    }


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
    analysis_df, info = build_univariate_re_analysis_frame(definition)
    dropped_studies = info["dropped_studies"]
    n_before_single_administration = info["n_before_single_administration"]
    unique_studies = info["unique_studies"]
    n_studies = len(unique_studies)
    n_subjects = info["n_subjects"]
    n_repeated_subjects = info["n_repeated_subjects"]

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
    if n_subjects is not None:
        data_rows.extend(
            [
                ("Subjects", n_subjects),
                ("Subjects with repeated observations", n_repeated_subjects),
            ]
        )
    if definition.one_observation_per_subject:
        data_rows.append(
            (
                "Single-administration sensitivity",
                f"{n_before_single_administration} -> {len(analysis_df)} rows",
            )
        )
    if definition.min_study_observations:
        data_rows.append(
            (
                f"Studies dropped (<{definition.min_study_observations} obs)",
                ", ".join(dropped_studies) if dropped_studies else "none",
            )
        )
    if "sex" in analysis_df.columns and "subject_id" in analysis_df.columns:
        children = analysis_df.drop_duplicates(["study", "subject_id"])["sex"]
        data_rows.append(
            (
                "Children by sex (girls / boys / unrecorded)",
                f"{int((children == 'F').sum())} / {int((children == 'M').sum())} / "
                f"{int(children.isna().sum())}",
            )
        )
    key_value_table("Data", data_rows)
    dataframe_table(study_counts, title="Observations per study")
    dataframe_table(desc, title="Descriptive statistics")

    X_obs = np.asarray(analysis_df["age"], dtype=float).reshape(-1, 1)
    y_values = np.asarray(analysis_df[y_col], dtype=float)
    # Validate BEFORE the integer cast: NumPy's cast truncates silently, so a
    # post-cast bound cannot catch 810.9 or -0.1, which truncate into range. This is
    # the range check this column gets on the fit path (#236, #240).
    # `build_univariate_re_model` below repeats it, because that builder is also
    # re-entered on simulated frames this stage never sees; the repeat is a no-op
    # here. Note it is `build_univariate_re_model`, not `common.build_model` --
    # that one belongs to the plain univariate engine, and naming it here is what
    # hid the gap.
    require_valid_counts(y_values, y_col, definition.n_trials)
    y_obs = y_values.astype(int)

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
    context: UnivariateREContext, definition: UnivariateModelDefinition
):
    """Pipeline stage: construct the model, then write its build report."""
    details = build_model_graph(context, definition)
    report_model_build(context, details)


def build_model_graph(
    context: UnivariateREContext,
    definition: UnivariateModelDefinition,
) -> BuildReport:
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
    build_report = BuildReport()
    config = context.model_config
    analysis_df = context.analysis_df

    y_col = definition.outcome.value

    X_obs = np.asarray(analysis_df["age"], dtype=float).reshape(-1, 1)
    # Validated here, on the pre-cast float, and not only in the prepare stage:
    # `recovery/simulate.py` re-enters this builder twice on a *simulated* frame
    # that no prepare stage has seen (`set_model_data(context.model_data, frame)`
    # then `build_stage(context)`), and VG11/VG12 are recovery targets. Three of the
    # six engines already validate inside their builders for this reason -- the
    # bivariate-RE one in `observation_arrays`, the joint one in `build`. On the
    # normal fit path the prepare stage has already run the same guard on the same
    # column, so this is a no-op there.
    y_values = np.asarray(analysis_df[y_col], dtype=float)
    require_valid_counts(y_values, y_col, definition.n_trials)
    y_obs = y_values.astype(int)
    study_codes = np.asarray(analysis_df["study_code"], dtype=int)

    n = len(X_obs)
    n_trials = context.model_data.n_trials
    n_studies = int(study_codes.max()) + 1
    use_subject_re = bool(definition.use_subject_re)
    if use_subject_re:
        subject_codes = np.asarray(analysis_df["subject_code"], dtype=int)
        n_subjects = int(subject_codes.max()) + 1
    else:
        subject_codes = None
        n_subjects = 0

    # Sex (#324): girls +1/2, boys -1/2, unrecorded 0, resolved per child.
    sex_sigma = sex_covariate.sex_effect_sigma(definition)
    x_sex = (
        sex_contrast_codes(analysis_df, allow_unknown=True)
        if sex_sigma is not None
        else None
    )

    # A child seen once contributes its effect to one likelihood term, so that
    # effect can be integrated out exactly instead of sampled. `partition` is
    # None unless the definition asks for it, and every branch below reads that
    # rather than the flag, so the sampled graph is untouched when it is off.
    # See models.subject_marginal.
    marginalisation = (
        getattr(definition, "singleton_marginalisation", None)
        if use_subject_re
        else None
    )
    partition = (
        partition_subject_rows(subject_codes) if marginalisation is not None else None
    )

    # Proposal A1 (#240 item 1): the child-effect scale varies log-linearly in age
    # between the dispersion block's anchors, with dispersion held flat in age, so
    # the age variation the constant loading routes into `kappa` is given to the
    # child scale instead. Resolved through the shared plan, which already reads
    # an `AgeVaryingSubjectScale` on `tau_subject_sigma`. Under the variance
    # partition the young-anchor scale is the partition's own `tau_subject`, so
    # the variant moves one thing: the ratio.
    age_varying = (
        subject_effects.resolve(definition)[UNIVARIATE_OUTCOME].age_varying
        if use_subject_re
        else None
    )
    if age_varying is not None and marginalisation is not None:
        raise ValueError(
            "An age-varying subject scale cannot be combined with singleton "
            "marginalisation: the quadrature integrates a constant child scale."
        )

    # Per-study age slopes (#240 item 5), measured in years from the GP anchor
    # age, or from the slope anchors' midpoint where the GP is not anchored.
    study_slope_sigma = getattr(definition, "study_age_slope_sigma", None)
    study_slope_reference = (
        float(definition.gp_anchor_age_months)
        if definition.gp_anchor_age_months is not None
        else float(np.mean(config.slope_anchors))
    )

    # Range validation happens before the integer cast, above, and in the prepare
    # stage -- never after it, where the cast has already truncated. Until
    # 2026-08-31 this engine used the weaker `require_integral_counts` and carried a
    # post-cast bounds check, so that check WAS load-bearing while the visually
    # identical ones in the bivariate engines were dead. All six engines now call
    # the same guard function, `build_utils.require_valid_counts`.

    # Standardise ages
    X_obs_mean, X_obs_std, X_obs_z = standardize_ages(X_obs)

    build_rows: list[tuple[str, object]] = [
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
    ]
    if use_subject_re:
        build_rows.extend(
            [
                ("Subject random intercept", True),
                ("Number of subjects", n_subjects),
            ]
        )
    if partition is not None:
        build_rows.extend(partition.summary_rows())
        build_rows.append(("Quadrature nodes", marginalisation.n_nodes))
    if x_sex is not None:
        build_rows.append(
            (
                "Sex contrast rows (girls / boys / unrecorded)",
                f"{int((x_sex > 0).sum())} / {int((x_sex < 0).sum())} / "
                f"{int((x_sex == 0).sum())}",
            )
        )
    if age_varying is not None:
        build_rows.append(
            ("Age-varying subject scale anchors (months)", age_varying.anchor_ages)
        )
    if study_slope_sigma is not None:
        build_rows.append(
            (
                "Study age slopes (per year, from months)",
                f"HalfNormal({study_slope_sigma:g}), from {study_slope_reference:g}",
            )
        )
    build_report.add_table("Build configuration", build_rows)

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
        *kappa_anchor_derived_rows(config, X_obs_mean=X_obs_mean, X_obs_std=X_obs_std),
    ]
    if anchor_g:
        derived_rows.append(("GP anchor age (months)", f"{anchor_age_months:g}"))
    build_report.add_table("Derived quantities", derived_rows)

    # Slice indices
    i_obs0, i_obs1 = grids.i_obs
    i_plot0, i_plot1 = grids.i_plot
    i_query0, i_query1 = grids.i_query

    coords = {
        "all_id": np.arange(n_all),
        "obs_id": np.arange(n),
        "plot_id": np.arange(n_plot),
        "query_id": np.arange(n_query),
        "study_id": np.arange(n_studies),
        "x_dim": np.arange(1),
    }
    if use_subject_re:
        # Marginalisation leaves explicit effects for the repeat-measured
        # children only, labelled by the child codes they keep, so a reader can
        # still say which child each effect belongs to.
        if partition is not None:
            coords["repeat_subject_id"] = partition.repeat_labels
        else:
            coords["subject_id"] = np.arange(n_subjects)

    with pm.Model(coords=coords) as model_pm:
        # ---- Data ----

        X_all_z_data = pm.Data("X_all_z", X_all_z, dims=("all_id", "x_dim"))

        _ = pm.Data("X_obs", X_obs.flatten(), dims=("obs_id",))
        _ = pm.Data("X_plot", X_plot.flatten(), dims=("plot_id",))
        _ = pm.Data("X_query", X_query.flatten(), dims=("query_id",))

        study_obs = pm.Data("study_obs", study_codes, dims=("obs_id",))
        if use_subject_re:
            subject_obs = pm.Data("subject_obs", subject_codes, dims=("obs_id",))

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
            grid=GPGrid.from_age_grids(
                grids,
                sa_z=slope_age_a_z,
                sb_z=slope_age_b_z,
                ell_low_z=ell_low_z,
                ell_high_z=ell_high_z,
                M=M,
                L=L,
            ),
            store_deterministic=True,
            latent_name="f_all",
            # getattr: the field lives on the joint definition classes only. The
            # 2026-08-04 mean-extrapolation fix was applied to the joint models
            # and never reached the univariate ones, which are exactly the models
            # where `eta` still presses its prior (VG01, VG03, VG11, VG12). See
            # notes/202608042030-q-mean-extrapolation.md.
            clamp_above_hi=getattr(definition, "clamp_mean_above_hi_anchor", False),
            anchor_idx=i_anchor if anchor_g else None,
            n_obs=n,
        )

        # ============================================================
        # Study-level random intercepts (non-centred, sum-to-zero)
        # ============================================================

        tau = pm.HalfNormal("tau", sigma=definition.tau_study_sigma)
        delta = zero_sum_study_offsets(
            "delta",
            scale=tau,
            n_studies=n_studies,
            centred=getattr(definition, "centred_study_re", False),
        )

        # The variance partition, when enabled, produces the subject scale here and
        # the young dispersion anchor further down, from one shared budget. It is
        # built before both so neither block can be written without the other.
        partition_excess_young = None
        if use_subject_re and config.has_variance_partition:
            tau_subject, partition_excess_young = build_variance_partition(
                config.variance_partition_total_dist,
                config.variance_partition_share_dist,
                reference_proportion=config.variance_partition_reference_proportion,
                subject_scale_name="tau_subject",
            )
        elif config.has_variance_partition:
            raise ValueError(
                "subject_variance_partition is set but use_subject_re is False; "
                "there is no subject scale for the budget to allocate."
            )

        tau_subject_of_z = None
        if use_subject_re and age_varying is not None:
            tau_subject_of_z, tau_subject = build_subject_scale_of_z(
                age_varying,
                anchor_z=standardize_anchor_ages(
                    age_varying.anchor_ages, X_obs_mean=X_obs_mean, X_obs_std=X_obs_std
                ),
                name="tau_subject",
                tau_young=tau_subject if partition_excess_young is not None else None,
            )
            delta_subject_raw = pm.Normal(
                "delta_subject_raw", mu=0.0, sigma=1.0, dims="subject_id"
            )
            # Stored at the young anchor, as the bivariate engine stores its A1
            # offsets, so `delta_subject` keeps its name and dims.
            _ = pm.Deterministic(
                "delta_subject", tau_subject * delta_subject_raw, dims="subject_id"
            )
            subject_shift = (
                tau_subject_of_z(X_all_z_data[i_obs0:i_obs1, 0])
                * delta_subject_raw[subject_obs]
            )
        elif use_subject_re:
            if partition_excess_young is None:
                tau_subject = pm.HalfNormal(
                    "tau_subject", sigma=definition.tau_subject_sigma
                )
            subject_dim = "subject_id" if partition is None else "repeat_subject_id"
            delta_subject_raw = pm.Normal(
                "delta_subject_raw", mu=0.0, sigma=1.0, dims=subject_dim
            )
            delta_subject = pm.Deterministic(
                "delta_subject",
                tau_subject * delta_subject_raw,
                dims=subject_dim,
            )
            if partition is None:
                subject_shift = delta_subject[subject_obs]
            else:
                # An exact zero on the rows whose effect the likelihood
                # integrates out, so f_obs there is the population-and-study
                # prediction rather than a child-specific one.
                subject_shift = zero_padded_subject_shift(delta_subject, partition)
        else:
            subject_shift = 0.0

        # ============================================================
        # Observation-level quantities (with study shift)
        # ============================================================

        # Add study shift to obs-level predictor only
        f_obs_re = f_all[i_obs0:i_obs1] + delta[study_obs] + subject_shift
        # Per-study age slopes (#240 item 5), emitted only when the definition
        # sets the field so every other graph is unchanged. Zero-sum, like the
        # intercepts, so the population slope keeps its meaning.
        if study_slope_sigma is not None:
            tau_slope = pm.HalfNormal("tau_slope", sigma=study_slope_sigma)
            delta_slope = zero_sum_study_offsets(
                "delta_slope", scale=tau_slope, n_studies=n_studies
            )
            years_from_reference = (X_obs.flatten() - study_slope_reference) / 12.0
            f_obs_re = f_obs_re + delta_slope[study_obs] * years_from_reference
        # Sex covariate (#324), emitted only when the definition sets it so every
        # other graph is unchanged. Added before `f_obs` is stored, so the
        # marginalised likelihood's `mu` carries it as the explicit one does.
        if x_sex is not None:
            x_sex_data = pm.Data("x_sex", x_sex, dims=("obs_id",))
            beta_sex = pm.Normal("beta_sex", mu=0.0, sigma=sex_sigma)
            f_obs_re = f_obs_re + beta_sex * x_sex_data

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
        _ = pm.Deterministic("z_obs", X_all_z_data[i_obs0:i_obs1, 0], dims=("obs_id",))
        _ = pm.Deterministic(
            "z_plot", X_all_z_data[i_plot0:i_plot1, 0], dims=("plot_id",)
        )
        _ = pm.Deterministic(
            "z_query", X_all_z_data[i_query0:i_query1, 0], dims=("query_id",)
        )
        # A1's scale on the reporting grids, read by the new-child predictive and
        # beside `kappa` in the report, as the bivariate engine writes it.
        if tau_subject_of_z is not None:
            _ = pm.Deterministic(
                "tau_subject_plot",
                tau_subject_of_z(X_all_z_data[i_plot0:i_plot1, 0]),
                dims="plot_id",
            )
            _ = pm.Deterministic(
                "tau_subject_query",
                tau_subject_of_z(X_all_z_data[i_query0:i_query1, 0]),
                dims="query_id",
            )

        # ============================================================
        # Dispersion / overdispersion
        # ============================================================

        kappa_of_z = build_kappa_for_config(
            config,
            X_obs_mean=X_obs_mean,
            X_obs_std=X_obs_std,
            excess_young_value=partition_excess_young,
            hold_constant=age_varying is not None and age_varying.hold_kappa_constant,
        )

        kappa_obs = pm.Deterministic(
            "kappa_obs", kappa_of_z(X_all_z_data[i_obs0:i_obs1, 0]), dims="obs_id"
        )
        _ = pm.Deterministic(
            "kappa_plot", kappa_of_z(X_all_z_data[i_plot0:i_plot1, 0]), dims="plot_id"
        )
        _ = pm.Deterministic(
            "kappa_query",
            kappa_of_z(X_all_z_data[i_query0:i_query1, 0]),
            dims="query_id",
        )

        # ============================================================
        # Beta-Binomial likelihood
        # ============================================================

        if partition is None:
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
        else:
            # One observed variable over every row either way: a repeat-measured
            # row keeps the identical conditional Beta-Binomial density, and a
            # marginalised row carries the quadrature integral of it over the
            # child effect. The pointwise log_likelihood keeps its name, shape
            # and row order; what changes is what a marginalised row's entry
            # means (marginal, not conditional).
            _ = subject_marginal_betabinomial(
                "y_obs",
                mu=f_obs,
                kappa=kappa_obs,
                tau_subject=tau_subject,
                observed=y_obs,
                n_trials=n_trials,
                partition=partition,
                n_nodes=marginalisation.n_nodes,
                dims=("obs_id",),
                epsilon=EPSILON,
            )

    variables = pymc_utils.get_variables_dict(model_pm)

    context.set_model(model_pm, variables)
    return build_report


# ============================================================
# Posterior predictive sampling
# ============================================================


def sample_posterior_predictive_re(
    context: UnivariateREContext,
    definition: UnivariateModelDefinition,
) -> None:
    """Sample a coherent trajectory for one new child when subject REs are used."""
    if not definition.use_subject_re:
        # No child to draw, so no new-child counts by sex either; the summary
        # stage writes the by-sex population tables from the coefficient alone.
        # This is the `single-admin` sensitivity's path on a model with sex.
        _base_sample_posterior_predictive(context, definition)
        return

    f_plot = context.model_variables["f_plot"]
    f_query = context.model_variables["f_query"]
    kappa_plot = context.model_variables["kappa_plot"]
    kappa_query = context.model_variables["kappa_query"]
    tau_subject = context.model_variables["tau_subject"]
    beta_sex = context.model_variables.get("beta_sex")

    with context.model:
        if "tau_subject_query" in context.model_variables:
            # A1: one standard deviate per new child, scaled by the child scale at
            # each age, which is what the fitted children were given.
            new_subject_deviate = pm.Normal("_z_subject_marg", mu=0.0, sigma=1.0)
            plot_shift = new_subject_deviate * context.model_variables["tau_subject_plot"]
            new_subject_shift = (
                new_subject_deviate * context.model_variables["tau_subject_query"]
            )
        else:
            new_subject_shift = pm.Normal(
                "_delta_subject_marg", mu=0.0, sigma=tau_subject
            )
            plot_shift = new_subject_shift
        p_plot = pm.math.sigmoid(f_plot + plot_shift)
        p_query = pm.math.sigmoid(f_query + new_subject_shift)

        p_plot = pm.math.clip(p_plot, EPSILON, 1 - EPSILON)
        p_query = pm.math.clip(p_query, EPSILON, 1 - EPSILON)
        pm.BetaBinomial(
            "y_plot",
            n=context.model_data.n_trials,
            alpha=p_plot * kappa_plot,
            beta=(1 - p_plot) * kappa_plot,
            dims=("plot_id",),
        )
        pm.BetaBinomial(
            "y_query",
            n=context.model_data.n_trials,
            alpha=p_query * kappa_query,
            beta=(1 - p_query) * kappa_query,
            dims=("query_id",),
        )
        # By sex (#324): the same new child, drawn once above, as a girl and as a
        # boy. `y_query` is that child at contrast zero -- a child of unrecorded
        # sex, which is how the model treats every such row it was fitted to.
        # Reusing the one child-effect draw makes the two levels paired draw for
        # draw, so their difference carries no between-child noise.
        by_sex_names: list[str] = []
        if beta_sex is not None:
            for level, contrast in sex_covariate.SEX_LEVELS:
                p_level = pm.Deterministic(
                    f"p_query_subject_marginal_{level}",
                    pm.math.sigmoid(f_query + new_subject_shift + contrast * beta_sex),
                    dims=("query_id",),
                )
                p_level = pm.math.clip(p_level, EPSILON, 1 - EPSILON)
                pm.BetaBinomial(
                    f"y_query_{level}",
                    n=context.model_data.n_trials,
                    alpha=p_level * kappa_query,
                    beta=(1 - p_level) * kappa_query,
                    dims=("query_id",),
                )
                by_sex_names += [f"p_query_subject_marginal_{level}", f"y_query_{level}"]
        trace = pm.sample_posterior_predictive(
            context.trace,
            var_names=["y_plot", "y_query", "y_obs", *by_sex_names],
            extend_inferencedata=True,
            progressbar=sys.stdout.isatty(),
            random_seed=context.sampling.random_seed,
        )

    context.set_trace(trace)
    calibration_df = write_trace_calibration(
        trace,
        context.analysis_df,
        context.reporting.output_dir,
        ((definition.outcome.value, "y_obs", None),),
    )
    context.dataframes["posterior_predictive_calibration"] = calibration_df
    save_trace(trace, context.reporting.output_dir)
    context.set_model_samples(extract_model_samples(trace))


def posterior_summary_re(
    context: UnivariateREContext, definition: UnivariateModelDefinition
) -> None:
    """The shared query-age summary, then the by-sex tables where the model has sex.

    Written at fit time because the report reads them from the fit's output and
    ``--render-only`` does not rebuild summary tables, so a table added later would
    need a refit to appear.
    """
    posterior_summary(context)
    trace = context.trace
    beta_sex = sex_covariate.coefficient_draws(trace, "beta_sex")
    if beta_sex is None:
        return
    samples = context.model_samples
    n_trials = context.model_data.n_trials
    ci_prob = context.reporting.ci_prob
    max_age = getattr(context.model_config, "report_max_age_understood", None)
    f_query = extract_posterior(trace, "f_query", "query_id")
    population = {
        level: sex_covariate.shifted(f_query, beta_sex, contrast)
        for level, contrast in sex_covariate.SEX_LEVELS
    }
    output_dir = context.reporting.output_dir
    # A model without child effects -- a `single-admin` sensitivity -- draws no
    # new child, so its by-sex table carries the population columns only.
    drew_children = "y_query_girls" in trace.posterior_predictive
    context.dataframes["posterior_summary_by_sex"] = (
        sex_covariate.write_probability_by_sex(
            output_dir,
            "",
            samples.X_query,
            population,
            n_trials=n_trials,
            ci_prob=ci_prob,
            interval_kind=context.reporting.interval_kind,
            predictive={
                level: extract_posterior_predictive(trace, f"y_query_{level}", "query_id")
                for level, _ in sex_covariate.SEX_LEVELS
            }
            if drew_children
            else None,
            subject_marginal={
                level: extract_posterior_predictive_float(
                    trace, f"p_query_subject_marginal_{level}", "query_id"
                )
                for level, _ in sex_covariate.SEX_LEVELS
            }
            if drew_children
            else None,
            max_age_months=max_age,
        )
    )
    sex_covariate.write_differences(
        output_dir,
        [
            sex_covariate.difference_rows(
                samples.X_query,
                definition.outcome.value,
                population["girls"],
                population["boys"],
                n_trials=n_trials,
                ci_prob=ci_prob,
                max_age_months=max_age,
            )
        ],
    )
    sex_covariate.write_coefficients(output_dir, {"beta_sex": beta_sex}, ci_prob=ci_prob)


# ============================================================
# Fit orchestration
# ============================================================


def univariate_re_stages(
    definition: UnivariateModelDefinition,
) -> list[tuple[str, Callable[[UnivariateREContext], None]]]:
    """The ordered ``(stage name, stage fn)`` list for this engine's fit.

    Exposed separately from :func:`fit_univariate_re_model` so a caller can
    substitute a single stage and still run the identical pipeline — the
    parameter-recovery harness swaps stage 0 (data preparation) for a loader
    that injects a simulated analysis frame (see
    :mod:`vocab_growth.recovery.refit`).
    """
    y_col = definition.outcome.value
    outcome_label = definition.outcome_label

    return [
        (
            "Prepare data",
            lambda ctx: prepare_univariate_re_data(ctx, definition),
        ),
        (
            "Priors and hyperparameters",
            lambda ctx: configure_univariate_priors(ctx, definition),
        ),
        (
            "Model definition and initialisation",
            lambda ctx: build_univariate_re_model(ctx, definition),
        ),
        (
            "Prior predictive checks",
            lambda ctx: prior_predictive_checks(
                ctx, outcome_col=y_col, outcome_label=outcome_label
            ),
        ),
        ("Posterior sampling", sample),
        ("Diagnostics", diagnostics),
        (
            "Posterior predictions",
            lambda ctx: sample_posterior_predictive_re(ctx, definition),
        ),
        ("Posterior summary", lambda ctx: posterior_summary_re(ctx, definition)),
        (
            "Plots",
            lambda ctx: run_standard_plots(
                    ctx,
                    outcome_label=outcome_label,
                    quantity=reporting_ages.quantity_for_outcome(definition.outcome),
                ),
        ),
        ("Report", report),
    ]


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
    return run_fit_pipeline(config, definition, stages=univariate_re_stages(definition))
