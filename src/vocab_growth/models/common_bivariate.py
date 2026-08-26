# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Shared dataclasses and pipeline functions for the bivariate vocabulary growth
models (e.g. VG05, VG07-VG10, VG13).

Uses a production-ratio reparameterization:
    p_U(a) = sigmoid(f_U(a))
    q(a)   = sigmoid(h(a))       # fraction of understood words spoken
    p_S(a) = p_U(a) * q(a)       # enforces p_S <= p_U by construction
"""

import os
import sys
from dataclasses import dataclass

import dse_research_utils.math.constants as math_constants
import dse_research_utils.plot.styles as plot_styles
import dse_research_utils.statistics.descriptive as descriptive_stats
import dse_research_utils.statistics.models.data as model_data
import dse_research_utils.statistics.models.pymc_utils as pymc_utils
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import preliz as pz
import pymc as pm
import xarray as xr
from preliz.distributions.distributions import Continuous

import vocab_growth.data_utils as vocab_data_utils
import vocab_growth.intervals as intervals
import vocab_growth.plotting as plotting
import vocab_growth.posterior_analysis as posterior_analysis
import vocab_growth.reporting_ages as reporting_ages
from vocab_growth.fit_artifacts import save_trace
from vocab_growth.models import prior_child_checks
from vocab_growth.models.build_utils import (
    construct_age_grids,
    require_valid_counts,
    slope_anchor_logit_coeffs,
    standardize_ages,
    validate_ell_bounds,
)
from vocab_growth.models.calibration import write_trace_calibration
from vocab_growth.models.common import (
    AnchoredKappaPriors,
    BaseModelConfiguration,
    ModelFitContext,
    _configure_kappa_priors,
    _plot_and_print_dist,
    build_kappa_for_config,
    emit_monthly_summary,
    get_hsgp_hyperparams,
    kappa_anchor_derived_rows,
    render_model_graph,
    report,
    run_fit_pipeline,
    validate_kappa_fields,
)
from vocab_growth.models.common import diagnostics as _shared_diagnostics
from vocab_growth.models.common import sample as _shared_sample
from vocab_growth.models.definitions import BivariateModelDefinition, clamp_targets
from vocab_growth.models.gp_utils import GPGrid, trend_and_gp
from vocab_growth.models.likelihood_utils import (
    SPOKEN_FALLBACK_PAIRED_ONLY,
    nested_outcome_alpha_beta,
    nested_outcome_spec,
    resolve_fallback_treatment,
)
from vocab_growth.plotting import (
    _save_csv,
    plot_comprehension_production_gap,
    plot_production_rate,
)
from vocab_growth.posterior_analysis import (
    extract_posterior as _extract_posterior,
)
from vocab_growth.posterior_analysis import (
    extract_posterior_predictive as _extract_posterior_predictive,
)
from vocab_growth.reporting import (
    console,
    dataframe_table,
    heading,
    key_value_table,
)

EPSILON = math_constants.EPSILON


# ============================================================
# Bivariate-specific dataclasses
# ============================================================


@dataclass
class BivariateModelConfiguration(BaseModelConfiguration):
    """Configuration for the bivariate (understood + spoken) model.

    Each outcome's dispersion is specified in exactly one of two ways: the
    legacy ``kappa_min_*_dist`` / ``a_kappa_*_dist`` / ``b_kappa_mag_*_dist``
    triple, or ``kappa_anchored_*``. The two outcomes are independent — VG13
    anchors both, the DS joint models anchor neither — but neither may be
    half-specified. ``__post_init__`` rejects anything else.
    """

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

    # Kappa priors — understood (legacy form)
    kappa_min_u_dist: Continuous | None = None
    a_kappa_u_dist: Continuous | None = None
    b_kappa_mag_u_dist: Continuous | None = None

    # Kappa priors — spoken (legacy form)
    kappa_min_s_dist: Continuous | None = None
    a_kappa_s_dist: Continuous | None = None
    b_kappa_mag_s_dist: Continuous | None = None

    # Two-anchor dispersion priors, in place of the triples above
    kappa_anchored_u: AnchoredKappaPriors | None = None
    kappa_anchored_s: AnchoredKappaPriors | None = None

    # Reporting only — the age at which understood and q stop being reported.
    report_max_age_understood: int | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        validate_kappa_fields(self, suffixes=("_u", "_s"))


@dataclass
class BivariateModelSamples:
    """Posterior and predictive samples from the bivariate model.

    Plot- and query-grid quantities only: the observation-level posterior
    (``f_u_obs``, ``p_u_obs``, ``h_obs``, ``q_obs``, ``f_s_obs``, ``p_s_obs``,
    ``z_obs``) used to be extracted as well, at ``n_obs x n_samples`` each, and
    nothing read it; since 2026-08-23 the sampler does not store those
    variables at all (:func:`vocab_growth.fit_artifacts.sampled_variable_names`).
    """

    # Shared age grids
    X_obs: np.ndarray
    """Observed ages in months, shape (n,)."""
    X_plot: np.ndarray
    """Ages in months for the plot points, shape (n_plot,)."""
    X_query: np.ndarray
    """Ages in months for the query points, shape (n_query,)."""

    X_plot_z: np.ndarray
    """Standardized ages for the plot points, shape (n_plot, n_samples)."""
    X_query_z: np.ndarray
    """Standardized ages for the query points, shape (n_query, n_samples)."""

    # Understood (U) samples
    f_u_plot: np.ndarray
    f_u_query: np.ndarray
    p_u_plot: np.ndarray
    p_u_query: np.ndarray
    p_u_query_subject_marginal: np.ndarray
    p_u_plot_subject_marginal: np.ndarray | None
    y_u_obs: np.ndarray
    y_u_plot: np.ndarray
    y_u_query: np.ndarray
    kappa_u_plot: np.ndarray
    kappa_u_query: np.ndarray

    # Production rate (q) samples
    h_plot: np.ndarray
    h_query: np.ndarray
    q_plot: np.ndarray
    q_query: np.ndarray

    # Spoken (S) samples (derived)
    f_s_plot: np.ndarray
    f_s_query: np.ndarray
    p_s_plot: np.ndarray
    p_s_query: np.ndarray
    p_s_query_subject_marginal: np.ndarray
    p_s_plot_subject_marginal: np.ndarray | None
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


def build_bivariate_analysis_frame(
    definition: BivariateModelDefinition,
) -> tuple[pd.DataFrame, dict]:
    """The exact prepared frame the bivariate engine fits, with no side effects.

    Split out of :func:`prepare_bivariate_data` so fitted-output validation can
    recompute the frame (and its exact hash) without a fit context — see
    :mod:`vocab_growth.analysis_frames` (issue #266 finding 1).
    """
    columns = ["age", "understood", "spoken"]
    if definition.exclude_us01_spoken_ceiling:
        columns.extend(["study", "survey_vocab_max"])
    df = vocab_data_utils.load_data(
        population=definition.population,
        columns=columns,
        sample_fraction=definition.sample_fraction,
        random_seed=definition.random_seed,
        # TD language scope is part of the model graph; DS ignores it.
        languages=getattr(
            definition, "td_languages", vocab_data_utils.ENGLISH_LANGUAGES
        ),
        include_implausible_production=definition.include_implausible_production,
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
    return analysis_df, {"ceiling_rows_excluded": ceiling_rows_excluded}


def prepare_bivariate_data(
    context: BivariateContext,
    definition: BivariateModelDefinition,
):
    """Load and prepare data for a bivariate model from its definition."""
    analysis_df, info = build_bivariate_analysis_frame(definition)
    ceiling_rows_excluded = info["ceiling_rows_excluded"]

    desc = descriptive_stats.describe_all(
        analysis_df[["age", "understood", "spoken"]], alpha=0.05
    )

    n = len(analysis_df)
    n_u = int(analysis_df["understood"].notna().sum())
    n_s = int(analysis_df["spoken"].notna().sum())
    n_both = int(
        (analysis_df["understood"].notna() & analysis_df["spoken"].notna()).sum()
    )

    counts: list[tuple[str, object]] = [
        ("Total observations", n),
        ("Understood observed", n_u),
        ("Spoken observed", n_s),
        ("Both observed", n_both),
        ("Understood only", n_u - n_both),
        ("Spoken only", n_s - n_both),
    ]
    if definition.exclude_us01_spoken_ceiling:
        counts.append(("us_01 WS-ceiling rows excluded", ceiling_rows_excluded))
    if definition.include_implausible_production:
        # No age bound: this engine's load_data call above passes none, so the
        # reported count has to be taken over the same frame or it misstates what
        # the fit actually reinstated.
        counts.append((
            "us_01 implausible production reinstated",
            vocab_data_utils.count_reinstated_implausible_production(),
        ))
    key_value_table("Observation counts", counts)
    dataframe_table(desc, title="Descriptive statistics")

    # Create a BinomialModelData for the context interface (using understood as primary)
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

    # --- Cross-lag coefficient prior (VG16) ---
    # Emitted as its own artefact so the report can put a prior figure beside
    # the beta_lag posterior — the shared trajectory prior predictives cannot
    # isolate this term (issue #242). A full effect-scale prior predictive
    # (the prior translated into q shifts over the empirical x_lag range)
    # remains registered follow-up work in #242.
    if getattr(definition, "use_cross_lag", False):
        heading("Cross-lag coefficient prior", style="bold cyan")
        beta_lag_dist = pz.Normal(
            mu=definition.beta_lag_mu, sigma=definition.beta_lag_sigma
        )
        _plot_and_print_dist(context, beta_lag_dist, "beta_lag_dist")

    # --- Kappa priors — understood ---
    heading("Kappa priors — understood", style="bold cyan")
    kappa_u_fields = _configure_kappa_priors(context, definition.kappa_u, "_u")

    # --- Kappa priors — spoken ---
    heading("Kappa priors — spoken", style="bold cyan")
    kappa_s_fields = _configure_kappa_priors(context, definition.kappa_s, "_s")

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
        n_plot=definition.n_plot,
        ages_query=definition.ages_query,
        report_max_age_understood=definition.report_max_age_understood,
        # Kappa — understood and spoken, each in whichever form it carries
        **kappa_u_fields,
        **kappa_s_fields,
    )

    context.set_model_config(config)


# ============================================================
# Model building
# ============================================================


def build_model(
    context: BivariateContext,
    definition: BivariateModelDefinition,
):
    """Build the bivariate PyMC model."""
    config = context.model_config

    analysis_df = context.analysis_df

    # Observation masks
    has_u = analysis_df["understood"].notna().values
    has_s = analysis_df["spoken"].notna().values

    X_obs = np.asarray(analysis_df["age"], dtype=float).reshape(-1, 1)
    n_trials = context.model_data.n_trials
    y_u_values = np.asarray(analysis_df.loc[has_u, "understood"], dtype=float)
    # Validate BEFORE the integer cast, exactly as the RE engine does: the cast
    # truncates silently, so the post-cast bounds checks below cannot catch
    # 810.9 or -0.1, which truncate into range (#236, #240).
    require_valid_counts(y_u_values, "understood", n_trials)
    y_u_observed = y_u_values.astype(int)

    idx_u = np.where(has_u)[0]

    n = len(X_obs)
    n_u = len(y_u_observed)
    spoken_spec = nested_outcome_spec(
        analysis_df,
        parent_col="understood",
        outcome_col="spoken",
        n_trials=n_trials,
    )
    if not np.array_equal(spoken_spec.indices, np.flatnonzero(has_s)):
        raise ValueError("Spoken likelihood rows do not match the observed-data mask.")
    # The mask check above runs against the unfiltered spec, so it still tests
    # what it was written to test under every treatment.
    spoken_fallback = resolve_fallback_treatment(definition)
    n_fallback_dropped = 0
    if spoken_fallback == SPOKEN_FALLBACK_PAIRED_ONLY:
        n_fallback_dropped = spoken_spec.n_marginal
        spoken_spec = spoken_spec.conditional_only()
    y_s_observed = spoken_spec.observed
    idx_s = spoken_spec.indices
    n_s = spoken_spec.n_observed
    # The stored spoken mask must mark the LIKELIHOOD rows: paired-only drops
    # the marginal fallback rows above, and calibration / extraction / LOO all
    # read ``obs_s_mask`` as "the rows y_s_obs covers" (issue #266 finding 3).
    # Under every other treatment this equals ``has_s`` exactly.
    has_s_likelihood = np.zeros(n, dtype=bool)
    has_s_likelihood[idx_s] = True

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

    key_value_table(
        "Build configuration",
        [
            ("Total observations", n),
            ("Understood observed", n_u),
            ("Spoken observed", n_s),
            ("Spoken conditional on understood", spoken_spec.n_conditional),
            ("Spoken marginal fallback", spoken_spec.n_marginal),
            ("Spoken fallback treatment", spoken_fallback),
            ("Spoken fallback rows dropped", n_fallback_dropped),
            ("Spoken > understood violations", spoken_spec.n_parent_violations),
            ("n_trials", n_trials),
            ("Age mean (months)", X_obs_mean),
            ("Age std (months)", X_obs_std),
            ("Slope anchors (months)", config.slope_anchors),
            ("Length-scale range (months)", config.ell_months_range),
            ("Query ages (months)", config.ages_query),
        ],
    )

    # Plot / query grids (standardised), stacked for 'free' predictions — see
    # models.build_utils.construct_age_grids.
    grids = construct_age_grids(
        X_obs,
        X_obs_z,
        X_obs_mean=X_obs_mean,
        X_obs_std=X_obs_std,
        n_plot=config.n_plot,
        ages_query=config.ages_query,
        slope_anchors=config.slope_anchors,
        gp_domain_months=definition.gp_domain_months,
    )
    X_plot = grids.X_plot
    X_query = grids.X_query
    X_all_z = grids.X_all_z
    n_plot = grids.n_plot
    n_query = grids.n_query
    n_all = grids.n_all

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

    key_value_table(
        "Derived quantities",
        [
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

        # Store masks and indices as constant data for extraction. The spoken
        # mask marks the likelihood rows (paired-only drops fallback rows), not
        # every observed-spoken row — see the derivation above (issue #266).
        _ = pm.Data("obs_u_mask", has_u.astype(int), dims=("obs_id",))
        _ = pm.Data("obs_s_mask", has_s_likelihood.astype(int), dims=("obs_id",))
        s_likelihood_n = pm.Data(
            "s_likelihood_n", spoken_spec.trials, dims=("obs_s_id",)
        )
        s_is_conditional = pm.Data(
            "s_is_conditional",
            spoken_spec.is_conditional.astype(int),
            dims=("obs_s_id",),
        )

        # Shared trend + HSGP builder (gp_utils); graph byte-identical to the
        # inlined form (stores g_u/f_u_all and g_q/h_all + the slope/intercept/ell).
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
            clamp_above_hi=_clamp_q,
        )

        # ============================================================
        # Derived quantities: p_U, q, p_S
        # ============================================================
        # Full-grid (n_all,) quantities are kept as plain tensors rather than
        # stored Deterministics: only their obs/plot/query slices below are
        # extracted, so materialising the full grid for every posterior draw
        # would waste a large amount of trace memory. (Matches the memory
        # discipline in common_bivariate_re.py and common_trivariate.py.)

        p_u_all = pm.math.sigmoid(f_u_all)
        q_all = pm.math.sigmoid(h_all)
        p_s_all = p_u_all * q_all

        # f_S derived for diagnostics/plotting
        p_s_all_clip = pm.math.clip(p_s_all, EPSILON, 1 - EPSILON)
        f_s_all = pm.math.log(p_s_all_clip) - pm.math.log(1 - p_s_all_clip)

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

        q_obs = pm.Deterministic("q_obs", q_all[i_obs0:i_obs1], dims=("obs_id",))
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

        kappa_u_of_z = build_kappa_for_config(
            config, X_obs_mean=X_obs_mean, X_obs_std=X_obs_std, suffix="_u"
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
            config, X_obs_mean=X_obs_mean, X_obs_std=X_obs_std, suffix="_s"
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
# Sample extraction
# ============================================================


def extract_model_samples(trace: xr.DataTree) -> BivariateModelSamples:
    """Extract model samples into a structured format for plotting and reporting."""

    # Understood
    f_u_plot = _extract_posterior(trace, "f_u_plot", "plot_id")
    f_u_query = _extract_posterior(trace, "f_u_query", "query_id")

    p_u_plot = _extract_posterior(trace, "p_u_plot", "plot_id")
    p_u_query = _extract_posterior(trace, "p_u_query", "query_id")

    kappa_u_plot = _extract_posterior(trace, "kappa_u_plot", "plot_id")
    kappa_u_query = _extract_posterior(trace, "kappa_u_query", "query_id")

    # Production rate
    h_plot = _extract_posterior(trace, "h_plot", "plot_id")
    h_query = _extract_posterior(trace, "h_query", "query_id")

    q_plot = _extract_posterior(trace, "q_plot", "plot_id")
    q_query = _extract_posterior(trace, "q_query", "query_id")

    # Spoken (derived)
    f_s_plot = _extract_posterior(trace, "f_s_plot", "plot_id")
    f_s_query = _extract_posterior(trace, "f_s_query", "query_id")

    p_s_plot = _extract_posterior(trace, "p_s_plot", "plot_id")
    p_s_query = _extract_posterior(trace, "p_s_query", "query_id")

    kappa_s_plot = _extract_posterior(trace, "kappa_s_plot", "plot_id")
    kappa_s_query = _extract_posterior(trace, "kappa_s_query", "query_id")

    # Observed data — expand to full obs_id length with NaN where unobserved
    obs_u_mask = np.array(trace.constant_data["obs_u_mask"].values, dtype=bool)
    obs_s_mask = np.array(trace.constant_data["obs_s_mask"].values, dtype=bool)
    n_obs = len(obs_u_mask)

    y_u_obs_raw = np.array(trace.observed_data["y_u_obs"].values, dtype=float)
    if int(obs_u_mask.sum()) != y_u_obs_raw.shape[0]:
        raise ValueError(
            f"obs_u_mask count ({int(obs_u_mask.sum())}) does not match observed "
            f"y_u_obs length ({y_u_obs_raw.shape[0]}); stored mask and likelihood "
            "rows are misaligned (issue #67)."
        )
    y_u_obs = np.full(n_obs, np.nan)
    y_u_obs[obs_u_mask] = y_u_obs_raw

    y_s_obs_raw = np.array(trace.observed_data["y_s_obs"].values, dtype=float)
    if int(obs_s_mask.sum()) != y_s_obs_raw.shape[0]:
        raise ValueError(
            f"obs_s_mask count ({int(obs_s_mask.sum())}) does not match observed "
            f"y_s_obs length ({y_s_obs_raw.shape[0]}); stored mask and likelihood "
            "rows are misaligned (issue #67)."
        )
    y_s_obs = np.full(n_obs, np.nan)
    y_s_obs[obs_s_mask] = y_s_obs_raw

    # Posterior predictive
    y_u_plot = _extract_posterior_predictive(trace, "y_u_plot", "plot_id")
    y_u_query = _extract_posterior_predictive(trace, "y_u_query", "query_id")
    y_s_plot = _extract_posterior_predictive(trace, "y_s_plot", "plot_id")
    y_s_query = _extract_posterior_predictive(trace, "y_s_query", "query_id")
    p_u_query_subject_marginal = posterior_analysis.extract_posterior_predictive_float(
        trace, "p_u_query_subject_marginal", "query_id"
    )
    p_s_query_subject_marginal = posterior_analysis.extract_posterior_predictive_float(
        trace, "p_s_query_subject_marginal", "query_id"
    )
    # Absent from any trace whose posterior predictive predates the plot-grid
    # pair; the trajectory overlay is then simply not drawn.
    p_u_plot_subject_marginal = _optional_posterior_predictive(
        trace, "p_u_plot_subject_marginal", "plot_id"
    )
    p_s_plot_subject_marginal = _optional_posterior_predictive(
        trace, "p_s_plot_subject_marginal", "plot_id"
    )

    # Constant data
    X_obs = np.array(trace.constant_data["X_obs"].values)
    X_plot = np.array(trace.constant_data["X_plot"].values)
    X_query = np.array(trace.constant_data["X_query"].values)

    # Standardised ages
    X_plot_z = _extract_posterior(trace, "z_plot", "plot_id")
    X_query_z = _extract_posterior(trace, "z_query", "query_id")

    return BivariateModelSamples(
        X_obs=X_obs,
        X_plot=X_plot,
        X_query=X_query,
        X_plot_z=X_plot_z,
        X_query_z=X_query_z,
        f_u_plot=f_u_plot,
        f_u_query=f_u_query,
        p_u_plot=p_u_plot,
        p_u_query=p_u_query,
        p_u_query_subject_marginal=p_u_query_subject_marginal,
        p_u_plot_subject_marginal=p_u_plot_subject_marginal,
        y_u_obs=y_u_obs,
        y_u_plot=y_u_plot,
        y_u_query=y_u_query,
        kappa_u_plot=kappa_u_plot,
        kappa_u_query=kappa_u_query,
        h_plot=h_plot,
        h_query=h_query,
        q_plot=q_plot,
        q_query=q_query,
        f_s_plot=f_s_plot,
        f_s_query=f_s_query,
        p_s_plot=p_s_plot,
        p_s_query=p_s_query,
        p_s_query_subject_marginal=p_s_query_subject_marginal,
        p_s_plot_subject_marginal=p_s_plot_subject_marginal,
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


def prior_predictive_checks(context: BivariateContext, definition=None):
    """Run prior predictive checks.

    The three figures below are **population mean functions**: they set every
    random effect to zero and carry no count noise, so they test the mean and
    not the counts. That is the right check for the trend and the GP, and until
    2026-08-24 it was the only check any of these reports had -- which left the
    gap #233 named, that a child-effect model's prior figures contain no child
    and so cannot test the prior the model was added for.

    When ``definition`` is supplied and the model carries child effects,
    :mod:`vocab_growth.models.prior_child_checks` adds the complementary
    figures: unseen-child trajectories, nested Beta-Binomial counts, and (for a
    model that couples the two outcomes) the induced joint association. Those
    are computed in NumPy from these same draws, so they add no node to the
    graph. ``definition`` is optional because the engines that carry no child
    effects call this without one.
    """
    with context.model:
        # No ``mode="FAST_COMPILE"`` -- it makes this fixed-cost stage 20-40x
        # slower for the same draws. See the note at the same call in
        # ``common.py`` and notes/202608251100-prior-predictive-compile-mode.md.
        prior_samples = pm.sample_prior_predictive(
            draws=1000,
            random_seed=context.sampling.random_seed,
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

    fig = plotting.plot_prior_samples_ratio(
        prior_samples.constant_data["X_plot"].values,
        q_plot_samples.values,
        y_label="q(a) = p_S(a) / p_U(a)",
        filename="prior_samples_q",
        output_dir=context.reporting.output_dir,
    )
    context.plots["prior_samples_q"] = fig
    plt.close(fig)

    if definition is not None:
        written = prior_child_checks.run(context, definition)
        if written:
            console.print(
                "[dim]Child-level prior checks: " + ", ".join(written) + "[/dim]"
            )


# ``sample`` is engine-agnostic (identical pm.sample() call in every engine) —
# reuse the shared implementation from common.py rather than redefining it.
sample = _shared_sample


def diagnostics(context: BivariateContext, definition=None):
    """Run diagnostics on the posterior samples.

    Thin wrapper over the shared engine (common.py): bivariate reports
    per-outcome LOO-CV for the spoken/understood likelihoods. (It used to name
    ``kappa_u_obs``/``kappa_s_obs`` for the trace plot as well; an
    observation-sized variable never fitted under ArviZ's subplot cap, so they
    never rendered, and since 2026-08-23 the sampler does not store them.)

    Both per-outcome scores are leave-one-likelihood-term-out rather than
    leave-one-administration-out: the spoken likelihood's trial count is the
    same administration's observed understood count, so a held-out spoken term
    is scored conditional on that observed comprehension, and a held-out
    understood term leaves its own observed value in the spoken term's
    denominator (#266).

    **The pair plot is reordered** for any definition carrying a distinguishing
    child structure, so the parameters the model was added for fill the capped
    grid instead of falling off the end of model order (issue #233). The
    ordering comes from :func:`pair_plot_priority`, which returns an empty tuple
    for a model without one -- and then the reordering is not installed at all,
    so those pair plots are unchanged. This began as VG16's own reordering for
    ``beta_lag`` (#242) and is now general.

    A cross-lag definition (VG16, issue #242) additionally **suppresses
    understood LOO**. The lag predictor embeds earlier observed understood
    counts as fixed covariates, so leaving one understood likelihood term out
    does not remove that count from the later spoken terms it predicts -- the
    "held-out" score still conditions on the held-out outcome, and Pareto-k
    cannot detect the leak. Spoken LOO is retained but labelled for what it
    estimates: prediction of a spoken count conditional on the child's observed
    understood history, not unconditional new-observation prediction.
    """
    posterior_vars = set(context.trace.posterior.data_vars)
    priority = pair_plot_priority(definition)

    def _prioritise(names: list[str]) -> list[str]:
        seen: set[str] = set()
        ordered: list[str] = []
        for name in (*priority, *names):
            if name in posterior_vars and name not in seen:
                ordered.append(name)
                seen.add(name)
        return ordered

    if not getattr(definition, "use_cross_lag", False):
        _shared_diagnostics(
            context,
            loo_var_names=(
                ("y_s_obs", "words spoken"),
                ("y_u_obs", "words understood"),
            ),
            var_names_fn=_prioritise if priority else None,
        )
        return

    console.print(
        "[yellow]Understood LOO is not computed for this model: the cross-lag "
        "predictor embeds earlier observed understood counts, so a pointwise "
        "leave-one-understood-out score would still condition on the held-out "
        "count through later spoken terms (issue #242). The spoken score below "
        "is prediction conditional on the child's observed understood "
        "history.[/yellow]"
    )
    _shared_diagnostics(
        context,
        loo_var_names=(
            ("y_s_obs", "words spoken (conditional on observed understood history)"),
        ),
        var_names_fn=_prioritise,
    )


def _subject_scales(context: BivariateContext, name: str):
    """The A1 plot/query subject scales for ``name``, or ``(None, None)``.

    Their presence is what tells the predictive path that the subject scale is
    age-varying; a model of record emits neither and takes the scalar branch, so
    its predictive graph is unchanged.
    """
    plot = context.model_variables.get(f"{name}_plot")
    query = context.model_variables.get(f"{name}_query")
    if plot is None or query is None:
        return None, None
    return plot, query


def _child_slope_block(context: BivariateContext, name: str):
    """The VG19 ``(tau0, tau1, rho01)`` scalars for ``name``, or ``None``.

    Detected by the three names :func:`~vocab_growth.models.gp_utils.build_child_slope`
    emits. This must be checked **before** :func:`_subject_scales`: a slope model
    also has an age-varying between-child SD, but scaling one deviate by it —
    the A1 branch — would impose perfect rank correlation across age, which is
    exactly the constraint the slope exists to relax. Same curve, different
    children.
    """
    mv = context.model_variables
    tau0, tau1, rho = (
        mv.get(f"{name}_0"),
        mv.get(f"{name}_1"),
        mv.get(f"{name}_rho"),
    )
    if tau0 is None or tau1 is None or rho is None:
        return None
    return tau0, tau1, rho


def _child_slope_offsets(context: BivariateContext, definition):
    """``(age - ref) / 12`` on the plot and query grids, in years.

    Read from the model's own ``X_plot`` / ``X_query`` data rather than recomputed,
    so the predictive cannot drift from the grids the fit actually used.
    """
    ref = float(getattr(definition, "subject_slope_ref_age_months", 36.0) or 36.0)
    # From the model's own named vars, not `context.model_variables`:
    # `get_variables_dict` collects free RVs, deterministics and observed RVs, so
    # `pm.Data` grids are absent from it.
    model = context.model
    return (
        (model["X_plot"] - ref) / 12.0,
        (model["X_query"] - ref) / 12.0,
    )


def pair_plot_priority(definition) -> tuple[str, ...]:
    """The variables the pair plot must show for ``definition``, most important first.

    ArviZ caps a pair plot at ``floor(sqrt(plot.max_subplots))`` variables, so a
    grid built in model order fits about six -- and model order is the build
    order, which puts the mean-function and GP parameters first. Every parameter
    a child-effect model was *added for* therefore fell off the end: VG19's
    slope block, VG20's ``rho_uq``, VG22's factor scales. The captions in those
    reports tell the reader to inspect exactly those ridges, so the plot
    contradicted the text it was captioned with (#233).

    Ordering rather than filtering, so nothing is hidden -- the cap simply
    consumes the list from a different end. An empty tuple means "model order",
    which is what every model without a distinguishing child structure gets, and
    those pair plots are byte-identical to before.

    The names are read from the definition rather than the trace so the intent
    is declared by the model, not inferred from what happened to be sampled.
    """
    priority: list[str] = []

    if getattr(definition, "use_cross_lag", False):
        priority.append("beta_lag")

    # VG20: the single parameter the model exists to estimate.
    if getattr(definition, "subject_re_correlation_eta", None) is not None:
        priority.append("rho_uq")

    # VG22: the factor form emits rho_uq as a deterministic and carries a rate
    # scale per outcome. `subject_factor_corr` is deliberately absent -- a 4x4
    # matrix is 16 plot items and would consume the whole grid on its own.
    if getattr(definition, "subject_factor", None) is not None:
        priority += ["rho_uq", "tau_subj_u_1", "tau_subj_q_1"]

    # VG19: the intercept-and-rate block, whose two correlations are the part a
    # reader can actually test from an interval.
    for name in ("tau_subj_u", "tau_subj_q"):
        spec = getattr(definition, f"{name}_sigma", None)
        if getattr(spec, "tau1_sigma", None) is not None:
            priority += [f"{name}_1", f"{name}_rho", f"{name}_0"]

    if priority:
        priority += ["tau_subj_u", "tau_subj_q", "tau_u", "tau_q"]
    return tuple(dict.fromkeys(priority))


def _child_factor_block(context: BivariateContext):
    """VG22's loading matrix, or ``None``.

    Detected by the one name :func:`~vocab_growth.models.gp_utils.build_child_factor`
    emits for it. Checked **before** :func:`_child_slope_block` and
    :func:`_subject_scales`, both of which a factor model would otherwise match:
    it emits ``tau_subj_*_0`` and ``tau_subj_*_1`` like a slope model, and a
    ``tau_subj_*`` like a constant-offset one, but its unseen child is neither a
    2x2 draw nor a scaled deviate.
    """
    return context.model_variables.get("subject_factor_loadings")


def unseen_child_correlated_delta_q(delta_u_query, *, tau_subj_u, tau_subj_q, rho):
    """VG20's unseen child: the q deviate that goes with an already-drawn u one.

    Extracted from :func:`sample_posterior_predictive` so the correlated branch
    can be executed by a test rather than only reached (#233). The two branches
    beside it -- VG19's slope and VG22's factor -- were already functions; this
    one was inline, and the only automated check on it was that ``rho_uq`` is
    visible from the predictive path, which is a precondition rather than the
    behaviour.

    The construction is unchanged, ops and names included, so the graph is
    identical: ``z_u`` is recovered by dividing the existing logit-scale deviate
    by its own scale rather than introducing a standardised RV, and the single
    new variable keeps the name ``_z_subj_q_marg``.

    Each deviate keeps its own marginal SD; only their joint behaviour changes.
    """
    z_u_marg = delta_u_query / tau_subj_u
    z_q_marg = pm.Normal("_z_subj_q_marg", mu=0.0, sigma=1.0)
    return tau_subj_q * (rho * z_u_marg + pm.math.sqrt(1.0 - rho**2) * z_q_marg)


def _unseen_child_factor_deltas(context, definition, outcome: str):
    """One unseen child's factor draw, as plot/query logit offsets for ``outcome``.

    The **same** child has to serve both outcomes -- that coupling is the entire
    point of the factor form -- but the two outcomes are handled in separate
    branches further down. So the child's ``b = L z`` is created once, on
    whichever branch runs first, and read back by name on the second. Drawing a
    fresh ``z`` per outcome would silently restore the independence VG22 exists
    to remove.
    """
    model = context.model
    if "_b_factor_marg" in model.named_vars:
        b = model.named_vars["_b_factor_marg"]
    else:
        loadings = context.model_variables["subject_factor_loadings"]
        z = pm.Normal("_z_factor_marg", mu=0.0, sigma=1.0, dims="factor")
        b = pm.Deterministic(
            "_b_factor_marg", pm.math.dot(loadings, z), dims="child_effect4"
        )
    ref = float(definition.subject_factor.ref_age_months)
    d_plot = (model["X_plot"] - ref) / 12.0
    d_query = (model["X_query"] - ref) / 12.0
    i0, i1 = (0, 1) if outcome == "u" else (2, 3)
    return b[i0] + b[i1] * d_plot, b[i0] + b[i1] * d_query


def _unseen_child_slope_deltas(context, definition, name, tag):
    """One ``(b0, b1)`` pair per posterior draw, as plot/query logit offsets.

    The unseen child is drawn from the *same* 2x2 joint the model fitted, so its
    trajectory fans with age and can cross another child's — the property that
    distinguishes a random slope from A1's rank-one scaling. Two standard
    deviates per draw, pushed through the model's own Cholesky.
    """
    tau0, tau1, rho = _child_slope_block(context, name)
    d_plot, d_query = _child_slope_offsets(context, definition)
    z0 = pm.Normal(f"_z0_{tag}_marg", mu=0.0, sigma=1.0)
    z1 = pm.Normal(f"_z1_{tag}_marg", mu=0.0, sigma=1.0)
    b0 = tau0 * z0
    b1 = tau1 * (rho * z0 + pm.math.sqrt(1.0 - rho**2) * z1)
    return b0 + b1 * d_plot, b0 + b1 * d_query


def sample_posterior_predictive(context: BivariateContext, definition=None):
    """Sample from the posterior predictive distribution.

    For models with subject-level random intercepts (use_subject_re_u and/or
    use_subject_re_q), the y_*_plot / y_*_query nodes are constructed using
    probabilities from one freshly-sampled subject RE per posterior draw. That
    gives a coherent unseen-child trajectory across age rather than independent
    one-age marginals at each plot or query point.
    """
    n_trials = context.model_data.n_trials

    p_u_plot = context.model_variables["p_u_plot"]
    p_u_query = context.model_variables["p_u_query"]
    kappa_u_plot = context.model_variables["kappa_u_plot"]
    kappa_u_query = context.model_variables["kappa_u_query"]

    p_s_query = context.model_variables["p_s_query"]
    q_plot = context.model_variables["q_plot"]
    q_query = context.model_variables["q_query"]
    kappa_s_plot = context.model_variables["kappa_s_plot"]
    kappa_s_query = context.model_variables["kappa_s_query"]

    use_subject_re_u = bool(getattr(definition, "use_subject_re_u", False))
    use_subject_re_q = bool(getattr(definition, "use_subject_re_q", False))

    with context.model:
        # Subject-marginalised probabilities if subject REs are present.
        # delta_subj_*_marg are auxiliary scalar RVs sampled from the subject-RE
        # prior during sample_posterior_predictive. Reusing one scalar across
        # plot/query ages makes y_*_plot a coherent unseen-child trajectory.
        if use_subject_re_u:
            tau_subj_u = context.model_variables["tau_subj_u"]
            f_u_plot_var = context.model_variables["f_u_plot"]
            f_u_query_var = context.model_variables["f_u_query"]
            plot_scale, query_scale = _subject_scales(context, "tau_subj_u")
            if _child_factor_block(context) is not None:
                # VG22. Checked before the slope branch, which its parameter
                # names would otherwise match.
                delta_u_plot, delta_u_query = _unseen_child_factor_deltas(
                    context, definition, "u"
                )
            elif _child_slope_block(context, "tau_subj_u") is not None:
                # VG19. Checked first: a slope model has an age-varying spread
                # too, but its unseen child is a (b0, b1) pair rather than one
                # deviate scaled by a curve.
                delta_u_plot, delta_u_query = _unseen_child_slope_deltas(
                    context, definition, "tau_subj_u", "subj_u"
                )
            elif plot_scale is None:
                delta_u_marg = pm.Normal("_delta_subj_u_marg", mu=0.0, sigma=tau_subj_u)
                delta_u_plot = delta_u_query = delta_u_marg
            else:
                # Proposal A1: one standard deviate per draw, scaled by tau(age),
                # so the unseen child stays the *same* child across the grid while
                # the spread it is drawn from widens or narrows with age. It gets
                # its own name: `_delta_subj_u_marg` holds a deviate already on
                # the logit scale, and reusing that name for a standardised one
                # would put two different quantities under it depending on the
                # branch taken.
                z_child_u = pm.Normal("_z_subj_u_marg", mu=0.0, sigma=1.0)
                delta_u_plot = z_child_u * plot_scale
                delta_u_query = z_child_u * query_scale
            p_u_plot = pm.math.sigmoid(f_u_plot_var + delta_u_plot)
            p_u_query = pm.math.sigmoid(f_u_query_var + delta_u_query)

        if use_subject_re_q:
            tau_subj_q = context.model_variables["tau_subj_q"]
            h_plot_var = context.model_variables["h_plot"]
            h_query_var = context.model_variables["h_query"]
            plot_scale, query_scale = _subject_scales(context, "tau_subj_q")
            # The unseen child's two deviates must come from the SAME joint
            # distribution the model fitted. Until 2026-08-19 they were two
            # independent `pm.Normal` draws, so a model carrying `rho_uq`
            # estimated the correlation and then discarded it when building the
            # subject-marginal predictive -- the one quantity the correlation was
            # added to change. VG20's gate 3 read as "a correlation of +0.368
            # leaves the spoken intervals unchanged", which was this code path
            # asserting rho = 0 rather than a result. See #224.
            #
            # Same Cholesky construction as `common_bivariate_re.build_model_re`:
            # each deviate keeps its marginal SD and only their joint behaviour
            # changes. `rho_uq` is absent from every model without a
            # `subject_re_correlation_eta`, so the other six draw exactly as
            # before -- checked by test rather than asserted.
            #
            # `z_u` is recovered by dividing the existing logit-scale deviate by
            # its own scale instead of introducing a standardised RV, so no
            # variable is renamed and no model's graph gains or loses a node.
            # The age-varying branch below cannot carry a correlation at all:
            # `_resolve_subject_re_correlation` rejects that combination,
            # because a constant correlation between per-observation-scaled
            # deviates is not a model anyone means.
            rho_marg = context.model_variables.get("rho_uq")
            correlated = rho_marg is not None and use_subject_re_u
            if _child_factor_block(context) is not None:
                # VG22, reading back the same child the u side drew. `correlated`
                # is False by construction: the engine refuses `rho_uq` as a
                # field alongside a factor, and the `rho_uq` this model emits is
                # a deterministic element of the factor covariance rather than a
                # coupling to apply again here.
                delta_q_plot, delta_q_query = _unseen_child_factor_deltas(
                    context, definition, "q"
                )
            elif _child_slope_block(context, "tau_subj_q") is not None:
                # VG19, and `correlated` is False by construction here: the
                # engine refuses `rho_uq` alongside a slope, so there is no
                # cross-outcome coupling to carry.
                delta_q_plot, delta_q_query = _unseen_child_slope_deltas(
                    context, definition, "tau_subj_q", "subj_q"
                )
            elif plot_scale is None:
                if correlated:
                    delta_q_marg = unseen_child_correlated_delta_q(
                        delta_u_query,
                        tau_subj_u=context.model_variables["tau_subj_u"],
                        tau_subj_q=tau_subj_q,
                        rho=rho_marg,
                    )
                else:
                    delta_q_marg = pm.Normal(
                        "_delta_subj_q_marg", mu=0.0, sigma=tau_subj_q
                    )
                delta_q_plot = delta_q_query = delta_q_marg
            else:
                z_child_q = pm.Normal("_z_subj_q_marg", mu=0.0, sigma=1.0)
                delta_q_plot = z_child_q * plot_scale
                delta_q_query = z_child_q * query_scale
            q_plot = pm.math.sigmoid(h_plot_var + delta_q_plot)
            q_query = pm.math.sigmoid(h_query_var + delta_q_query)
            p_s_query = p_u_query * q_query
        elif use_subject_re_u:
            # Marginal U but not q: rebuild p_s from new p_u and original q.
            p_s_query = p_u_query * q_query

        pm.Deterministic(
            "p_u_query_subject_marginal", p_u_query, dims=("query_id",)
        )
        pm.Deterministic(
            "p_s_query_subject_marginal", p_s_query, dims=("query_id",)
        )
        # The same quantities on the plot grid, which is what a *trajectory*
        # needs. `y_*_plot` below is already a coherent unseen child -- one child
        # effect reused across the grid -- but it carries Beta-Binomial noise
        # drawn independently at each of the grid's points, and on VG20 that noise
        # moves a mean of ~90 words between adjacent points against ~1.3 for the
        # curve underneath it. Drawing the counts as a line gives scribble; these
        # are the expected trajectory the noise is scattered around.
        #
        # Stored rather than recomputed by the plotting code because the unseen
        # child differs by model -- a correlated pair under `rho_uq`, a (b0, b1)
        # pair for VG19, a factor block for VG22, one deviate scaled by tau(age)
        # under the age-varying scale -- and reproducing that outside this
        # function is how the correlation came to be silently dropped once
        # before (#224).
        pm.Deterministic(
            "p_u_plot_subject_marginal", p_u_plot, dims=("plot_id",)
        )
        pm.Deterministic(
            "p_s_plot_subject_marginal", p_u_plot * q_plot, dims=("plot_id",)
        )

        # Understood — plot
        p_u_plot_clip = pm.math.clip(p_u_plot, EPSILON, 1 - EPSILON)
        y_u_plot = pm.BetaBinomial(
            "y_u_plot",
            n=n_trials,
            alpha=p_u_plot_clip * kappa_u_plot,
            beta=(1 - p_u_plot_clip) * kappa_u_plot,
            dims=("plot_id",),
        )
        # Understood — query
        p_u_query_clip = pm.math.clip(p_u_query, EPSILON, 1 - EPSILON)
        y_u_query = pm.BetaBinomial(
            "y_u_query",
            n=n_trials,
            alpha=p_u_query_clip * kappa_u_query,
            beta=(1 - p_u_query_clip) * kappa_u_query,
            dims=("query_id",),
        )
        # Spoken — plot
        q_plot_clip = pm.math.clip(q_plot, EPSILON, 1 - EPSILON)
        pm.BetaBinomial(
            "y_s_plot",
            n=y_u_plot,
            alpha=q_plot_clip * kappa_s_plot,
            beta=(1 - q_plot_clip) * kappa_s_plot,
            dims=("plot_id",),
        )
        # Spoken — query
        q_query_clip = pm.math.clip(q_query, EPSILON, 1 - EPSILON)
        pm.BetaBinomial(
            "y_s_query",
            n=y_u_query,
            alpha=q_query_clip * kappa_s_query,
            beta=(1 - q_query_clip) * kappa_s_query,
            dims=("query_id",),
        )

        trace = pm.sample_posterior_predictive(
            context.trace,
            var_names=[
                "y_u_plot",
                "y_u_query",
                "p_u_query_subject_marginal",
                "p_u_plot_subject_marginal",
                "y_u_obs",
                "y_s_plot",
                "y_s_query",
                "p_s_query_subject_marginal",
                "p_s_plot_subject_marginal",
                "y_s_obs",
            ],
            extend_inferencedata=True,
            progressbar=sys.stdout.isatty(),
            random_seed=context.sampling.random_seed,
        )

    context.set_trace(trace)

    calibration_df = write_trace_calibration(
        trace,
        context.analysis_df,
        context.reporting.output_dir,
        (
            ("understood", "y_u_obs", "obs_u_mask"),
            ("spoken", "y_s_obs", "obs_s_mask"),
        ),
        strata={
            "spoken": (
                "s_is_conditional",
                "spoken (conditional)",
                "spoken (fallback)",
            )
        },
    )
    context.dataframes["posterior_predictive_calibration"] = calibration_df

    save_trace(trace, context.reporting.output_dir)

    sample_data = extract_model_samples(context.trace)
    context.set_model_samples(sample_data)


def posterior_summary(context: BivariateContext):
    """Compute and store the posterior summary tables at query ages."""
    samples = context.model_samples
    n_trials = context.model_data.n_trials
    ci_prob = context.reporting.ci_prob
    has_subject_re = any(
        name in context.model_variables for name in ("tau_subj_u", "tau_subj_q")
    )
    # Comprehension and production are not observed over the same age range, so
    # understood and q may report a shorter grid than spoken.
    report_max_u = context.model_config.report_max_age_understood

    # Understood summary
    summary_u = posterior_analysis.posterior_summary_table(
        samples.X_query,
        samples.p_u_query,
        samples.y_u_query,
        n_trials=n_trials,
        ci_prob=ci_prob,
    )
    if has_subject_re:
        summary_u = posterior_analysis.add_probability_estimand_columns(
            summary_u,
            samples.p_u_query,
            samples.p_u_query_subject_marginal,
            n_trials=n_trials,
            ci_prob=ci_prob,
        )
    summary_u = posterior_analysis.trim_reported_ages(summary_u, report_max_u)
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
        ci_prob=ci_prob,
    )
    if has_subject_re:
        summary_s = posterior_analysis.add_probability_estimand_columns(
            summary_s,
            samples.p_s_query,
            samples.p_s_query_subject_marginal,
            n_trials=n_trials,
            ci_prob=ci_prob,
        )
    dataframe_table(
        summary_s, title="Posterior summary — words spoken", show_index=False
    )
    context.dataframes["posterior_summary_s"] = summary_s
    summary_s.to_csv(
        os.path.join(context.reporting.output_dir, "posterior_summary_s.csv"),
        index=False,
    )

    # Production rate summary (equal-tailed; q is a bounded ratio)
    summary_q = intervals.summarise(
        samples.q_query, samples.X_query, name="q_query", outer=ci_prob, sample_axis=1
    ).rename(
        columns={
            "median": "q_median",
            "ci50_lo": "q_ci50_lo",
            "ci50_hi": "q_ci50_hi",
            "ci_lo": "q_ci_lo",
            "ci_hi": "q_ci_hi",
        }
    )
    summary_q = posterior_analysis.trim_reported_ages(summary_q, report_max_u)
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
    max_age_months_understood: float | None = None,
    max_age_months_spoken: float | None = None,
):
    """Plot both understood and spoken posterior predictive median trends on one figure.

    This figure carries **two** outcomes with different evidence, so it takes two
    caps rather than one: understood stops at its comprehension cap and spoken at
    its own, and each series is trimmed independently. A single
    ``max_age_months`` could only have been the narrower of the two, which would
    have thrown away the spoken tail the figure exists to show. See
    :mod:`vocab_growth.reporting_ages`.

    The companion CSV keeps one row per age out to the wider cap, and blanks the
    understood columns beyond the narrower one — so a reader can see that
    comprehension stops being reported rather than inferring it from a short row.
    """
    X_plot = samples.X_plot

    outer, inner = intervals.DEFAULT_CI_PROB, intervals.INNER_CI_PROB

    # Understood
    y_u_median = np.quantile(samples.y_u_plot, 0.50, axis=1)
    y_u_ci = intervals.bands(samples.y_u_plot, outer, "eti", sample_axis=1)
    y_u_ci50 = intervals.bands(samples.y_u_plot, inner, "eti", sample_axis=1)

    # Spoken
    y_s_median = np.quantile(samples.y_s_plot, 0.50, axis=1)
    y_s_ci = intervals.bands(samples.y_s_plot, outer, "eti", sample_axis=1)
    y_s_ci50 = intervals.bands(samples.y_s_plot, inner, "eti", sample_axis=1)

    ku = np.ones_like(X_plot, dtype=bool) if max_age_months_understood is None \
        else X_plot <= max_age_months_understood
    ks = np.ones_like(X_plot, dtype=bool) if max_age_months_spoken is None \
        else X_plot <= max_age_months_spoken

    fig, ax = plt.subplots(figsize=plot_styles.FIGSIZE_XL)

    # Understood bands
    ax.fill_between(X_plot[ku], y_u_ci[ku, 0], y_u_ci[ku, 1], alpha=0.15, color="C0")
    ax.fill_between(X_plot[ku], y_u_ci50[ku, 0], y_u_ci50[ku, 1], alpha=0.25, color="C0")
    ax.plot(X_plot[ku], y_u_median[ku], lw=3, color="C0", label="Words understood (median)")

    # Spoken bands
    ax.fill_between(X_plot[ks], y_s_ci[ks, 0], y_s_ci[ks, 1], alpha=0.15, color="C1")
    ax.fill_between(X_plot[ks], y_s_ci50[ks, 0], y_s_ci50[ks, 1], alpha=0.25, color="C1")
    ax.plot(X_plot[ks], y_s_median[ks], lw=3, color="C1", label="Words spoken (median)")

    # Observed data, each modality trimmed with its own curve.
    X_obs = samples.X_obs
    u_cap = np.inf if max_age_months_understood is None else max_age_months_understood
    s_cap = np.inf if max_age_months_spoken is None else max_age_months_spoken
    u_mask = ~np.isnan(samples.y_u_obs) & (X_obs <= u_cap)
    if u_mask.any():
        ax.scatter(X_obs[u_mask], samples.y_u_obs[u_mask], s=10, alpha=0.2, color="C0")
    s_mask = ~np.isnan(samples.y_s_obs) & (X_obs <= s_cap)
    if s_mask.any():
        ax.scatter(X_obs[s_mask], samples.y_s_obs[s_mask], s=10, alpha=0.2, color="C1")

    ax.set_xlabel("Age (months)")
    ax.set_ylabel("Word count")
    ax.legend(loc="upper left", frameon=True)
    ax.set_ylim(-20, n_trials + 50)

    if output_dir is not None and filename is not None:
        fig.savefig(os.path.join(output_dir, f"{filename}.png"), dpi=300)
        fig.savefig(os.path.join(output_dir, f"{filename}.svg"))
        blank_u = np.where(ku, 1.0, np.nan)
        blank_s = np.where(ks, 1.0, np.nan)
        _save_csv(pd.DataFrame({
            "age_months": X_plot,
            "understood_median": y_u_median * blank_u,
            "understood_ci50_lo": y_u_ci50[:, 0] * blank_u,
            "understood_ci50_hi": y_u_ci50[:, 1] * blank_u,
            "understood_ci_lo": y_u_ci[:, 0] * blank_u,
            "understood_ci_hi": y_u_ci[:, 1] * blank_u,
            "spoken_median": y_s_median * blank_s,
            "spoken_ci50_lo": y_s_ci50[:, 0] * blank_s,
            "spoken_ci50_hi": y_s_ci50[:, 1] * blank_s,
            "spoken_ci_lo": y_s_ci[:, 0] * blank_s,
            "spoken_ci_hi": y_s_ci[:, 1] * blank_s,
        })[ku | ks], output_dir, filename)

    return fig


def plot_understood_spoken_trajectory_intervals(
    samples: BivariateModelSamples,
    n_trials: int,
    output_dir: str | None = None,
    filename: str | None = None,
    max_age_months_understood: float | None = None,
    max_age_months_spoken: float | None = None,
):
    """Plot understood and spoken posterior predictive medians with 50% and 89% equal-tailed intervals.

    The bands summarise the posterior predictive draws (``y_u_plot`` / ``y_s_plot``),
    not the mean trajectory, and are equal-tailed per the house convention in
    :mod:`vocab_growth.intervals` — counts are not on the HDI short-list.

    Two caps, one per outcome, for the reason given in
    :func:`plot_understood_spoken_trajectory`.
    """
    X_plot = samples.X_plot
    outer, inner = intervals.DEFAULT_CI_PROB, intervals.INNER_CI_PROB
    pct = int(round(outer * 100))

    ku = np.ones_like(X_plot, dtype=bool) if max_age_months_understood is None \
        else X_plot <= max_age_months_understood
    ks = np.ones_like(X_plot, dtype=bool) if max_age_months_spoken is None \
        else X_plot <= max_age_months_spoken

    # Understood bands (equal-tailed)
    y_u_median = np.median(samples.y_u_plot, axis=1)
    y_u_ci = intervals.bands(samples.y_u_plot, outer, "eti", sample_axis=1)
    y_u_ci50 = intervals.bands(samples.y_u_plot, inner, "eti", sample_axis=1)

    # Spoken bands (equal-tailed)
    y_s_median = np.median(samples.y_s_plot, axis=1)
    y_s_ci = intervals.bands(samples.y_s_plot, outer, "eti", sample_axis=1)
    y_s_ci50 = intervals.bands(samples.y_s_plot, inner, "eti", sample_axis=1)

    fig, ax = plt.subplots(figsize=plot_styles.FIGSIZE_XL)

    # Understood bands
    ax.fill_between(X_plot[ku], y_u_ci[ku, 0], y_u_ci[ku, 1], alpha=0.15, color="C0", label=f"Words understood ({pct}% interval)")
    ax.fill_between(X_plot[ku], y_u_ci50[ku, 0], y_u_ci50[ku, 1], alpha=0.25, color="C0", label="Words understood (50% interval)")
    ax.plot(X_plot[ku], y_u_median[ku], lw=3, color="C0", label="Words understood (median)")

    # Spoken bands
    ax.fill_between(X_plot[ks], y_s_ci[ks, 0], y_s_ci[ks, 1], alpha=0.15, color="C1", label=f"Words spoken ({pct}% interval)")
    ax.fill_between(X_plot[ks], y_s_ci50[ks, 0], y_s_ci50[ks, 1], alpha=0.25, color="C1", label="Words spoken (50% interval)")
    ax.plot(X_plot[ks], y_s_median[ks], lw=3, color="C1", label="Words spoken (median)")

    ax.set_xlabel("Age (months)")
    ax.set_ylabel("Word count")
    ax.legend(loc="upper left", frameon=True)
    ax.set_ylim(-20, n_trials + 50)

    if output_dir is not None and filename is not None:
        fig.savefig(os.path.join(output_dir, f"{filename}.png"), dpi=300)
        fig.savefig(os.path.join(output_dir, f"{filename}.svg"))
        blank_u = np.where(ku, 1.0, np.nan)
        blank_s = np.where(ks, 1.0, np.nan)
        _save_csv(pd.DataFrame({
            "age_months": X_plot,
            "understood_median": y_u_median * blank_u,
            "understood_ci50_lo": y_u_ci50[:, 0] * blank_u,
            "understood_ci50_hi": y_u_ci50[:, 1] * blank_u,
            "understood_ci_lo": y_u_ci[:, 0] * blank_u,
            "understood_ci_hi": y_u_ci[:, 1] * blank_u,
            "spoken_median": y_s_median * blank_s,
            "spoken_ci50_lo": y_s_ci50[:, 0] * blank_s,
            "spoken_ci50_hi": y_s_ci50[:, 1] * blank_s,
            "spoken_ci_lo": y_s_ci[:, 0] * blank_s,
            "spoken_ci_hi": y_s_ci[:, 1] * blank_s,
        })[ku | ks], output_dir, filename)

    return fig


def plot_production_rate_by_understood(
    samples: BivariateModelSamples,
    n_trials: int,
    ci_prob: float = intervals.DEFAULT_CI_PROB,
    interval_kind: intervals.IntervalKind = "eti",
    output_dir: str | None = None,
    filename: str | None = None,
    max_age_months: float | None = None,
):
    """Plot population production ratio q against population expected words understood.

    **What this is, and what it is not (issue #233).** Both axes are read off the
    *population* curves at zero study and zero child effects: ``p_u_plot`` and
    ``q_plot``. The x value at a plotted point is the population median expected
    comprehension AT SOME AGE, and the y value is the population conversion ratio
    AT THAT SAME AGE. The curve therefore describes how the two population
    trajectories move together as children get older -- a developmental-stage
    relationship -- and NOT the conditional quantity ``E[q | understood = U]``
    for a child who happens to understand U words.

    The two differ whenever children vary, and here they differ in a known
    direction. A child observed above the population comprehension curve carries
    a positive understood child effect, and under VG20's ``rho_uq`` = +0.368 a
    positive conversion effect with it, so the genuine conditional expectation
    rises with U more steeply than this curve does. Nothing here conditions the
    child effects on observed comprehension or uses ``rho_uq`` at all. Computing
    the conditional version means integrating the joint child-effect posterior
    through the understood Beta-Binomial likelihood, which is a separate output.

    ``max_age_months`` is essential here rather than cosmetic. The x axis is age
    *reparameterised* by expected comprehension, so without the cap the curve
    silently extends past ``report_max_age_understood`` -- the age at which the
    age-space plot of the same quantity stops. Worse, the mean is clamped above
    the upper slope anchor, so expected understood almost stops growing there and
    the x axis compresses hard: a gentle drift in ``q`` over the extrapolated tail
    is then drawn as a near-vertical step, which reads as a discovery about
    vocabulary rather than an artefact of the transform.
    """
    p_u_plot = samples.p_u_plot  # (n_plot, n_samples)
    q_plot = samples.q_plot  # (n_plot, n_samples)

    if max_age_months is not None:
        keep = np.asarray(samples.X_plot) <= max_age_months
        p_u_plot = p_u_plot[keep, :]
        q_plot = q_plot[keep, :]

    x_words = np.median(p_u_plot, axis=1) * n_trials
    q_median = np.median(q_plot, axis=1)
    q_ci = intervals.bands(q_plot, ci_prob, interval_kind, sample_axis=1)
    q_ci50 = intervals.bands(q_plot, intervals.INNER_CI_PROB, interval_kind, sample_axis=1)
    pct = int(round(ci_prob * 100))

    fig, ax = plt.subplots(figsize=plot_styles.FIGSIZE_XL)

    ax.fill_between(
        x_words,
        q_ci[:, 0],
        q_ci[:, 1],
        alpha=0.20,
        label=f"{pct}% interval",
    )
    ax.fill_between(
        x_words,
        q_ci50[:, 0],
        q_ci50[:, 1],
        alpha=0.30,
        label="50% interval",
    )
    ax.plot(x_words, q_median, lw=3, label="Median q")

    ax.set_xlabel("Population expected words understood (by age)")
    ax.set_ylabel("q = p_S / p_U")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left", frameon=True)
    ax.set_title("Population production ratio by developmental stage")

    if output_dir is not None and filename is not None:
        fig.savefig(os.path.join(output_dir, f"{filename}.png"), dpi=300)
        fig.savefig(os.path.join(output_dir, f"{filename}.svg"))
        _save_csv(pd.DataFrame({
            "words_understood": x_words,
            "q_median": q_median,
            "ci50_lo": q_ci50[:, 0],
            "ci50_hi": q_ci50[:, 1],
            "ci_lo": q_ci[:, 0],
            "ci_hi": q_ci[:, 1],
        }), output_dir, filename)

    return fig


def plot_production_rate_predictive(
    samples: BivariateModelSamples,
    output_dir: str | None = None,
    filename: str | None = None,
    max_age_months: float | None = None,
):
    """Plot posterior predictive spoken/understood count ratio with 50% and 89% intervals.

    Uses posterior predictive counts (y_s / y_u). Samples where y_u == 0 are
    excluded per age point before computing summary statistics.

    ``max_age_months`` applies the same comprehension cap as the population-level
    :func:`plot_production_rate`. Without it the predictive twin of a capped plot
    runs further than the plot it mirrors, and the pair disagree about where the
    evidence ends.
    """
    X_plot = samples.X_plot
    y_u = samples.y_u_plot  # (n_plot, n_samples)
    y_s = samples.y_s_plot

    if max_age_months is not None:
        keep = np.asarray(X_plot) <= max_age_months
        X_plot = X_plot[keep]
        y_u = y_u[keep, :]
        y_s = y_s[keep, :]
    outer, inner = intervals.DEFAULT_CI_PROB, intervals.INNER_CI_PROB
    pct = int(round(outer * 100))

    n_ages = y_u.shape[0]
    ratio_median = np.empty(n_ages)
    ratio_ci = np.empty((n_ages, 2))
    ratio_ci50 = np.empty((n_ages, 2))

    for i in range(n_ages):
        mask = y_u[i] > 0
        if mask.sum() < 10:
            ratio_median[i] = np.nan
            ratio_ci[i] = np.nan
            ratio_ci50[i] = np.nan
            continue
        ratio_i = y_s[i, mask] / y_u[i, mask]
        ratio_median[i] = np.median(ratio_i)
        ratio_ci[i] = intervals.interval_1d(ratio_i, outer, "eti")
        ratio_ci50[i] = intervals.interval_1d(ratio_i, inner, "eti")

    fig, ax = plt.subplots(figsize=plot_styles.FIGSIZE_XL)

    ax.fill_between(
        X_plot,
        ratio_ci[:, 0],
        ratio_ci[:, 1],
        alpha=0.20,
        label=f"{pct}% interval",
    )
    ax.fill_between(
        X_plot,
        ratio_ci50[:, 0],
        ratio_ci50[:, 1],
        alpha=0.30,
        label="50% interval",
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
            "ci50_lo": ratio_ci50[:, 0],
            "ci50_hi": ratio_ci50[:, 1],
            "ci_lo": ratio_ci[:, 0],
            "ci_hi": ratio_ci[:, 1],
        }), output_dir, filename)

    return fig


def plot_understood_vs_spoken(
    samples: BivariateModelSamples,
    n_trials: int,
    n_draws: int = 200,
    output_dir: str | None = None,
    filename: str | None = None,
    max_age_months: float | None = None,
):
    """Plot posterior expected words understood (x) vs words spoken (y) as spaghetti curves.

    ``max_age_months`` caps the curve at the comprehension reporting age: the x
    axis is expected comprehension, so every point past that age is extrapolation
    plotted on a compressed axis (see :func:`plot_production_rate_by_understood`).
    """
    X_plot = np.asarray(samples.X_plot)
    E_u = samples.p_u_plot * n_trials  # (n_plot, n_samples)
    E_s = samples.p_s_plot * n_trials

    if max_age_months is not None:
        keep = X_plot <= max_age_months
        X_plot = X_plot[keep]
        E_u = E_u[keep, :]
        E_s = E_s[keep, :]

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
            "age_months": X_plot,
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
    max_age_months: float | None = None,
):
    """Plot posterior predictive words understood (x) vs words spoken (y) as spaghetti curves.

    ``max_age_months`` caps the curve at the comprehension reporting age, matching
    the population-level :func:`plot_understood_vs_spoken`.
    """
    X_plot = np.asarray(samples.X_plot)
    y_u = samples.y_u_plot  # (n_plot, n_samples)
    y_s = samples.y_s_plot

    if max_age_months is not None:
        keep = X_plot <= max_age_months
        X_plot = X_plot[keep]
        y_u = y_u[keep, :]
        y_s = y_s[keep, :]

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
            "age_months": X_plot,
            "understood_median": y_u_median,
            "spoken_median": y_s_median,
        }), output_dir, filename)

    return fig


def plot_spoken_given_understood(
    samples: BivariateModelSamples,
    n_trials: int,
    ci_prob: float = intervals.DEFAULT_CI_PROB,
    interval_kind: intervals.IntervalKind = "eti",
    output_dir: str | None = None,
    filename: str | None = None,
    max_age_months: float | None = None,
):
    """Words spoken implied by the POPULATION conversion ratio, by age (issue #112, Q1).

    A fan of lines, one per representative age, each with slope the population
    ``q(a)`` from ``q_query`` and shaded by that ratio's posterior interval. The
    dashed y = x line is the ceiling: a child cannot say more distinct words than
    they understand.

    **The estimand is population-level, and the line is not a conditional
    expectation (issue #233).** ``q_query`` is evaluated at zero study and zero
    child effects, so ``U * q(a)`` is "U words converted at the rate a typical
    child of age a converts at" -- not ``E[spoken | understood = U, age a]``.
    The nested likelihood does give ``E[S | U, a, child] = U * q(a, child)``
    exactly, but the child's own conversion effect is missing from ``q(a)``, and
    it is not independent of U: a child understanding more words than typical for
    their age has a positive understood child effect, which under VG20's
    ``rho_uq`` = +0.368 comes with a positive conversion effect. So the genuine
    conditional line is steeper than this one at high U and shallower at low U,
    and this plot understates the spread besides, showing only the uncertainty in
    ``q(a)`` and none of the between-child variation in it.

    Read a line as "what the population rate implies at this comprehension
    level", and read the caveat with it wherever it is published.
    """
    q_query = samples.q_query  # (n_query, n_samples)
    ages = np.asarray(samples.X_query)  # (n_query,)

    # q is a ratio of comprehension, so ages past the comprehension reporting age
    # are extrapolation. Drop them before selecting the fan, or the fan spends one
    # of its five lines on an age the model declines to report q for elsewhere.
    if max_age_months is not None:
        keep = ages <= max_age_months
        ages = ages[keep]
        q_query = q_query[keep, :]

    # A few representative ages spanning the query range (keeps the fan legible).
    n_age = len(ages)
    sel = np.unique(np.linspace(0, n_age - 1, num=min(5, n_age)).round().astype(int))

    understood = np.linspace(0, n_trials, 100)

    fig, ax = plt.subplots(figsize=plot_styles.FIGSIZE_XL)
    rows = []
    for k, j in enumerate(sel):
        q_samps = q_query[j, :]
        q_med = float(np.median(q_samps))
        q_lo, q_hi = intervals.interval_1d(q_samps, ci_prob, interval_kind)
        color = f"C{k}"
        ax.fill_between(
            understood, understood * q_lo, understood * q_hi, alpha=0.15, color=color
        )
        ax.plot(
            understood,
            understood * q_med,
            lw=2.5,
            color=color,
            label=f"{int(round(float(ages[j])))} mo (q={q_med:.2f})",
        )
        rows.append(pd.DataFrame({
            "age_months": int(round(float(ages[j]))),
            "words_understood": understood,
            "spoken_median": understood * q_med,
            "spoken_ci_lo": understood * q_lo,
            "spoken_ci_hi": understood * q_hi,
        }))

    ax.plot(
        [0, n_trials], [0, n_trials], ls="--", lw=1, color="grey",
        label="spoken = understood",
    )

    ax.set_xlabel("Words understood")
    ax.set_ylabel("Words spoken at the population rate")
    ax.set_xlim(0, n_trials)
    ax.set_ylim(0, n_trials)
    ax.set_title("Words spoken implied by the population conversion ratio")
    ax.legend(loc="upper left", frameon=True, title="Age")

    if output_dir is not None and filename is not None:
        fig.savefig(os.path.join(output_dir, f"{filename}.png"), dpi=300)
        fig.savefig(os.path.join(output_dir, f"{filename}.svg"))
        _save_csv(pd.concat(rows, ignore_index=True), output_dir, filename)

    return fig


# ============================================================
# Shared bivariate plotting pipeline
# ============================================================


def _expected_counts(probabilities, n_trials: int):
    """Unseen-child probabilities as expected counts, or ``None`` if absent.

    The plot works in counts, and the stored subject-marginal quantity is a
    probability on the fixed item scale.
    """
    if probabilities is None:
        return None
    return np.asarray(probabilities) * n_trials


def _optional_column(analysis_df: pd.DataFrame, mask, name: str) -> pd.Series | None:
    """``analysis_df[mask, name]``, or ``None`` where the column is not carried.

    The frame's columns depend on the definition -- a model without subject
    random effects has no ``subject_key`` -- and the trajectory overlay degrades
    to the plain scatter rather than failing when one is absent.
    """
    if name not in analysis_df.columns:
        return None
    return analysis_df.loc[mask, name]


def _optional_posterior_predictive(trace, name: str, dim: str) -> np.ndarray | None:
    """``extract_posterior_predictive_float``, or ``None`` where the name is absent.

    A trace written before a predictive variable existed must still load: the
    features that read it degrade, rather than the whole extractor failing.
    """
    group = getattr(trace, "posterior_predictive", None)
    if group is None or name not in group.data_vars:
        return None
    return posterior_analysis.extract_posterior_predictive_float(trace, name, dim)


def _run_bivariate_outcome_plots(
    samples: BivariateModelSamples,
    y_plot: np.ndarray,
    y_query: np.ndarray,
    f_plot: np.ndarray,
    p_plot: np.ndarray,
    kappa_plot: np.ndarray,
    kappa_query: np.ndarray,
    x_obs: pd.Series,
    y_obs: pd.Series,
    n_trials: int,
    ci_prob: float,
    output_dir: str,
    suffix: str,
    outcome_label: str,
    y_label: str,
    max_age_months: float | None = None,
    subject_ids: pd.Series | None = None,
    form_max: pd.Series | None = None,
    trajectory_samples: np.ndarray | None = None,
):
    """Run the standard per-outcome plotting pipeline for a bivariate model.

    ``max_age_months`` is this outcome's reporting cap, resolved by the caller
    from :mod:`vocab_growth.reporting_ages`. Every artefact below is a pure
    function of one outcome, so they all take the same cap -- which is the point
    of resolving it once here rather than at each call.
    """
    emit_monthly_summary(
        output_dir=output_dir,
        X_plot=samples.X_plot,
        p_plot=p_plot,
        y_plot=y_plot,
        X_obs=x_obs,
        n_trials=n_trials,
        ci_prob=ci_prob,
        suffix=suffix,
        outcome_label=outcome_label.lower(),
        y_label=y_label,
        max_age_months=max_age_months,
    )

    plotting.plot_posterior_predictive_count_distributions_by_query_age(
        X_query=samples.X_query,
        y_query=y_query,
        n_trials=n_trials,
        ci_prob=ci_prob,
        output_dir=output_dir,
        filename=f"posterior_predictive_count_distributions_{suffix}",
        x_label=f"{outcome_label} (count)",
        max_age_months=max_age_months,
    )

    plotting.plot_posterior_predictive_pmf(
        samples.X_query,
        y_query,
        n_trials,
        output_dir=output_dir,
        filename=f"posterior_predictive_pmf_{suffix}",
        x_label=f"{outcome_label} (count)",
        max_age_months=max_age_months,
    )

    plotting.plot_posterior_predictive_cdf(
        samples.X_query,
        y_query,
        n_trials,
        output_dir=output_dir,
        filename=f"posterior_predictive_cdf_{suffix}",
        x_label=f"{outcome_label} (count)",
        max_age_months=max_age_months,
    )

    plotting.plot_posterior_predictive_median_trend(
        samples.X_plot,
        y_plot,
        x_obs,
        y_obs,
        output_dir=output_dir,
        filename=f"posterior_predictive_median_trend_{suffix}",
        y_label=y_label,
        max_age_months=max_age_months,
        subject_ids=subject_ids,
        form_max=form_max,
        trajectory_samples=trajectory_samples,
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
        max_age_months=max_age_months,
        subject_ids=subject_ids,
        form_max=form_max,
        trajectory_samples=trajectory_samples,
    )

    plotting.plot_expected_learning_rate(
        samples.X_plot,
        f_plot,
        n_trials=n_trials,
        ci_prob=ci_prob,
        output_dir=output_dir,
        filename=f"expected_learning_rate_{suffix}",
        y_label=f"Estimated {outcome_label.lower()} gain per month",
        max_age_months=max_age_months,
    )

    plotting.plot_expected_learning_rate(
        samples.X_plot,
        f_plot,
        n_trials=n_trials,
        ci_prob=ci_prob,
        smooth=True,
        savgol_window_length=15,
        savgol_polyorder=3,
        smooth_intervals=True,
        output_dir=output_dir,
        filename=f"expected_learning_rate_{suffix}_smoothed",
        y_label=f"Estimated {outcome_label.lower()} gain per month",
        max_age_months=max_age_months,
    )

    plotting.plot_posterior_kappa(
        samples.X_plot,
        kappa_plot,
        samples.X_query,
        kappa_query,
        n_trials=n_trials,
        ci_prob=ci_prob,
        output_dir=output_dir,
        filename=f"posterior_kappa_{suffix}",
        max_age_months=max_age_months,
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
        max_age_months_understood=reporting_ages.max_age_for(
            context.model_config, reporting_ages.ReportedQuantity.UNDERSTOOD
        ),
        max_age_months_spoken=reporting_ages.max_age_for(
            context.model_config, reporting_ages.ReportedQuantity.SPOKEN
        ),
    )
    context.plots["joint_trajectory"] = fig
    plt.close(fig)

    # ---- Joint trajectory interval plot ----

    fig = plot_understood_spoken_trajectory_intervals(
        samples,
        n_trials=context.model_data.n_trials,
        output_dir=context.reporting.output_dir,
        filename="joint_trajectory_intervals",
        max_age_months_understood=reporting_ages.max_age_for(
            context.model_config, reporting_ages.ReportedQuantity.UNDERSTOOD
        ),
        max_age_months_spoken=reporting_ages.max_age_for(
            context.model_config, reporting_ages.ReportedQuantity.SPOKEN
        ),
    )
    context.plots["joint_trajectory_intervals"] = fig
    plt.close(fig)

    # ---- Production rate q(a) ----

    fig = plot_production_rate(
        samples,
        ci_prob=context.reporting.ci_prob,
        output_dir=context.reporting.output_dir,
        filename="production_rate",
        max_age_months=context.model_config.report_max_age_understood,
    )
    context.plots["production_rate"] = fig
    plt.close(fig)

    # ---- Production rate by words understood ----

    fig = plot_production_rate_by_understood(
        samples,
        n_trials=definition.n_trials,
        ci_prob=context.reporting.ci_prob,
        output_dir=context.reporting.output_dir,
        filename="production_rate_by_understood",
        max_age_months=context.model_config.report_max_age_understood,
    )
    context.plots["production_rate_by_understood"] = fig
    plt.close(fig)

    # ---- Posterior predictive production rate ----

    fig = plot_production_rate_predictive(
        samples,
        output_dir=context.reporting.output_dir,
        filename="production_rate_predictive",
        max_age_months=context.model_config.report_max_age_understood,
    )
    context.plots["production_rate_predictive"] = fig
    plt.close(fig)

    # ---- Comprehension-production gap ----

    fig = plot_comprehension_production_gap(
        samples,
        n_trials=context.model_data.n_trials,
        ci_prob=context.reporting.ci_prob,
        output_dir=context.reporting.output_dir,
        filename="comprehension_production_gap",
        max_age_months=context.model_config.report_max_age_understood,
    )
    context.plots["comprehension_production_gap"] = fig
    plt.close(fig)

    # ---- Understood vs spoken ----

    fig = plot_understood_vs_spoken(
        samples,
        n_trials=context.model_data.n_trials,
        output_dir=context.reporting.output_dir,
        filename="understood_vs_spoken",
        max_age_months=context.model_config.report_max_age_understood,
    )
    context.plots["understood_vs_spoken"] = fig
    plt.close(fig)

    # ---- Posterior predictive understood vs spoken ----

    fig = plot_understood_vs_spoken_predictive(
        samples,
        n_trials=context.model_data.n_trials,
        output_dir=context.reporting.output_dir,
        filename="understood_vs_spoken_predictive",
        max_age_months=context.model_config.report_max_age_understood,
    )
    context.plots["understood_vs_spoken_predictive"] = fig
    plt.close(fig)

    # ---- Predicted spoken given understood, by age (issue #112, Q1) ----

    fig = plot_spoken_given_understood(
        samples,
        n_trials=context.model_data.n_trials,
        ci_prob=context.reporting.ci_prob,
        output_dir=context.reporting.output_dir,
        filename="spoken_given_understood",
        max_age_months=context.model_config.report_max_age_understood,
    )
    context.plots["spoken_given_understood"] = fig
    plt.close(fig)

    # ---- Per-outcome plots: understood ----

    _run_bivariate_outcome_plots(
        samples=samples,
        y_plot=samples.y_u_plot,
        y_query=samples.y_u_query,
        f_plot=samples.f_u_plot,
        p_plot=samples.p_u_plot,
        kappa_plot=samples.kappa_u_plot,
        kappa_query=samples.kappa_u_query,
        x_obs=analysis_df.loc[has_u, "age"],
        y_obs=analysis_df.loc[has_u, "understood"],
        subject_ids=_optional_column(analysis_df, has_u, "subject_key"),
        form_max=_optional_column(analysis_df, has_u, "survey_vocab_max"),
        trajectory_samples=_expected_counts(
            samples.p_u_plot_subject_marginal, context.model_data.n_trials
        ),
        n_trials=context.model_data.n_trials,
        ci_prob=context.reporting.ci_prob,
        output_dir=context.reporting.output_dir,
        suffix="u",
        outcome_label="Words understood",
        y_label="Predicted words understood",
        max_age_months=reporting_ages.max_age_for(
            context.model_config, reporting_ages.ReportedQuantity.UNDERSTOOD
        ),
    )

    # ---- Per-outcome plots: spoken ----

    _run_bivariate_outcome_plots(
        samples=samples,
        y_plot=samples.y_s_plot,
        y_query=samples.y_s_query,
        f_plot=samples.f_s_plot,
        p_plot=samples.p_s_plot,
        kappa_plot=samples.kappa_s_plot,
        kappa_query=samples.kappa_s_query,
        x_obs=analysis_df.loc[has_s, "age"],
        y_obs=analysis_df.loc[has_s, "spoken"],
        subject_ids=_optional_column(analysis_df, has_s, "subject_key"),
        form_max=_optional_column(analysis_df, has_s, "survey_vocab_max"),
        trajectory_samples=_expected_counts(
            samples.p_s_plot_subject_marginal, context.model_data.n_trials
        ),
        n_trials=context.model_data.n_trials,
        ci_prob=context.reporting.ci_prob,
        output_dir=context.reporting.output_dir,
        suffix="s",
        outcome_label="Words spoken",
        y_label="Predicted words spoken",
        max_age_months=reporting_ages.max_age_for(
            context.model_config, reporting_ages.ReportedQuantity.SPOKEN
        ),
    )


# ============================================================
# Fit orchestration
# ============================================================


def fit_bivariate_model(
    config: str,
    definition: BivariateModelDefinition,
) -> BivariateContext:
    """
    Shared fit pipeline for bivariate models (e.g. VG05, VG07-VG10, VG13).
    """
    return run_fit_pipeline(
        config,
        definition,
        stages=[
            ("Prepare data", lambda ctx: prepare_bivariate_data(ctx, definition)),
            (
                "Priors and hyperparameters",
                lambda ctx: configure_bivariate_priors(ctx, definition),
            ),
            (
                "Model definition and initialisation",
                lambda ctx: build_model(ctx, definition),
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
        ],
    )
