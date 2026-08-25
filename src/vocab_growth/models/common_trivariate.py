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
import vocab_growth.environment as local_env
import vocab_growth.intervals as intervals
import vocab_growth.plotting as plotting
import vocab_growth.posterior_analysis as posterior_analysis
import vocab_growth.reporting_ages as reporting_ages
from vocab_growth.fit_artifacts import save_trace
from vocab_growth.models.build_utils import (
    construct_age_grids,
    require_integral_counts,
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
    render_model_graph,
    report,
    run_fit_pipeline,
    validate_kappa_fields,
)
from vocab_growth.models.common import diagnostics as _shared_diagnostics
from vocab_growth.models.common import sample as _shared_sample
from vocab_growth.models.definitions import TrivariateModelDefinition, clamp_targets
from vocab_growth.models.gp_utils import (
    GPGrid,
    tent_and_gp,
    trend_and_gp,
)
from vocab_growth.models.likelihood_utils import nested_outcome_spec
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
    dataframe_table,
    heading,
    key_value_table,
)

EPSILON = math_constants.EPSILON


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

    # Signed ratio (r) priors — three-anchor hump (young / peak / old)
    p_slope_low_sign_dist: Continuous
    p_slope_mid_sign_dist: Continuous
    p_slope_hi_sign_dist: Continuous
    ell_unit_sign_dist: Continuous
    eta_sign_dist: Continuous
    sign_anchor_ages: tuple[float, float, float]

    # Kappa priors — understood (legacy form)
    kappa_min_u_dist: Continuous | None = None
    a_kappa_u_dist: Continuous | None = None
    b_kappa_mag_u_dist: Continuous | None = None

    # Kappa priors — spoken (legacy form)
    kappa_min_s_dist: Continuous | None = None
    a_kappa_s_dist: Continuous | None = None
    b_kappa_mag_s_dist: Continuous | None = None

    # Kappa priors — signed (legacy form)
    kappa_min_sign_dist: Continuous | None = None
    a_kappa_sign_dist: Continuous | None = None
    b_kappa_mag_sign_dist: Continuous | None = None

    # Two-anchor dispersion priors, in place of the triples above. This engine
    # accepted only the legacy form until 2026-08-06, which is why VG14 still
    # carried it while VG10 and VG15 had migrated — the definition was not the
    # blocker, the engine was.
    kappa_anchored_u: AnchoredKappaPriors | None = None
    kappa_anchored_s: AnchoredKappaPriors | None = None
    kappa_anchored_sign: AnchoredKappaPriors | None = None

    # Reporting only — the age at which understood and q stop being reported.
    report_max_age_understood: int | None = None
    # Reporting only — the age at which signed quantities stop being reported.
    # Distinct from the comprehension cap: the sign-derived figures used to be
    # trimmed by that one, so a comprehension decision silently moved them.
    report_max_age_signed: int | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        validate_kappa_fields(self, suffixes=("_u", "_s", "_sign"))


@dataclass
class TrivariateModelSamples:
    """Posterior and predictive samples from the trivariate model.

    Plot- and query-grid quantities only; the observation-level posterior is no
    longer extracted (nothing read it) or stored by the sampler -- see
    :class:`vocab_growth.models.common_bivariate.BivariateModelSamples`.
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
    f_sign_plot: np.ndarray
    f_sign_query: np.ndarray
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

    df, sign_source_dropped = vocab_data_utils.mask_incomparable_signed_outcomes(
        df,
        include_signed_only=definition.include_uk01_signed,
    )

    analysis_df = df[["study", "age", "understood", "spoken", "signed"]].copy()

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
            (
                f"uk_01 signed-only dropped "
                f"(include_uk01_signed={definition.include_uk01_signed})",
                sign_source_dropped.get("uk_01", 0),
            ),
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

    # Three-anchor hump signed mean (young / peak / old): Beta priors on r at three
    # reference ages, interpolated as a tent meeting at the peak (gp_utils.tent_and_gp).
    p_slope_low_sign_dist = pz.Beta(
        alpha=definition.p_slope_low_sign_alpha, beta=definition.p_slope_low_sign_beta
    )
    _plot_and_print_dist(context, p_slope_low_sign_dist, "p_slope_low_sign_dist")

    p_slope_mid_sign_dist = pz.Beta(
        alpha=definition.p_slope_mid_sign_alpha, beta=definition.p_slope_mid_sign_beta
    )
    _plot_and_print_dist(context, p_slope_mid_sign_dist, "p_slope_mid_sign_dist")

    p_slope_hi_sign_dist = pz.Beta(
        alpha=definition.p_slope_hi_sign_alpha, beta=definition.p_slope_hi_sign_beta
    )
    _plot_and_print_dist(context, p_slope_hi_sign_dist, "p_slope_hi_sign_dist")

    # --- Kappa priors — understood ---
    heading("Kappa priors — understood", style="bold cyan")

    kappa_u_fields = _configure_kappa_priors(
        context, definition.kappa_u, "_u"
    )

    # --- Kappa priors — spoken ---
    heading("Kappa priors — spoken", style="bold cyan")

    kappa_s_fields = _configure_kappa_priors(
        context, definition.kappa_s, "_s"
    )

    # --- Kappa priors — signed ---
    heading("Kappa priors — signed", style="bold cyan")

    kappa_sign_fields = _configure_kappa_priors(
        context, definition.kappa_sign, "_sign"
    )

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
        # Signed rate (three-anchor hump mean)
        p_slope_low_sign_dist=p_slope_low_sign_dist,
        p_slope_mid_sign_dist=p_slope_mid_sign_dist,
        p_slope_hi_sign_dist=p_slope_hi_sign_dist,
        ell_unit_sign_dist=ell_unit_sign_dist,
        eta_sign_dist=eta_sign_dist,
        sign_anchor_ages=definition.sign_anchor_ages,
        # Dispersion, per outcome. Each dict carries whichever parameterisation
        # that outcome uses -- the legacy triple or `kappa_anchored_*` -- so the
        # three outcomes may differ, which VG14 now relies on (understood and
        # spoken anchored, signed still legacy).
        **kappa_u_fields,
        **kappa_s_fields,
        **kappa_sign_fields,
        n_plot=definition.n_plot,
        ages_query=definition.ages_query,
        report_max_age_understood=definition.report_max_age_understood,
        report_max_age_signed=definition.report_max_age_signed,
    )

    context.set_model_config(config)


# ============================================================
# Model building
# ============================================================


def build_model(
    context: TrivariateContext,
    definition: TrivariateModelDefinition,
):
    """Build the trivariate PyMC model."""
    config = context.model_config

    analysis_df = context.analysis_df

    # Observation masks
    has_u = analysis_df["understood"].notna().values
    has_s = analysis_df["spoken"].notna().values
    has_sign = analysis_df["signed"].notna().values

    X_obs = np.asarray(analysis_df["age"], dtype=float).reshape(-1, 1)
    y_u_values = np.asarray(analysis_df.loc[has_u, "understood"], dtype=float)
    require_integral_counts(y_u_values, "understood")
    y_u_observed = y_u_values.astype(int)

    idx_u = np.where(has_u)[0]

    n = len(X_obs)
    n_u = len(y_u_observed)
    n_trials = context.model_data.n_trials
    spoken_spec = nested_outcome_spec(
        analysis_df,
        parent_col="understood",
        outcome_col="spoken",
        n_trials=n_trials,
    )
    signed_spec = nested_outcome_spec(
        analysis_df,
        parent_col="understood",
        outcome_col="signed",
        n_trials=n_trials,
    )
    if not np.array_equal(spoken_spec.indices, np.flatnonzero(has_s)):
        raise ValueError("Spoken likelihood rows do not match the observed-data mask.")
    if not np.array_equal(signed_spec.indices, np.flatnonzero(has_sign)):
        raise ValueError("Signed likelihood rows do not match the observed-data mask.")
    y_s_observed = spoken_spec.observed
    y_sign_observed = signed_spec.observed
    idx_s = spoken_spec.indices
    idx_sign = signed_spec.indices
    n_s = spoken_spec.n_observed
    n_sign = signed_spec.n_observed

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
    X_obs_mean, X_obs_std, X_obs_z = standardize_ages(X_obs)

    key_value_table(
        "Build configuration",
        [
            ("Total observations", n),
            ("Understood observed", n_u),
            ("Spoken observed", n_s),
            ("Spoken conditional on understood", spoken_spec.n_conditional),
            ("Spoken marginal fallback", spoken_spec.n_marginal),
            ("Spoken > understood violations", spoken_spec.n_parent_violations),
            ("Signed observed", n_sign),
            ("Signed conditional on understood", signed_spec.n_conditional),
            ("Signed marginal fallback", signed_spec.n_marginal),
            ("Signed > understood violations", signed_spec.n_parent_violations),
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
        s_likelihood_n = pm.Data(
            "s_likelihood_n", spoken_spec.trials, dims=("obs_s_id",)
        )
        s_is_conditional = pm.Data(
            "s_is_conditional",
            spoken_spec.is_conditional.astype(int),
            dims=("obs_s_id",),
        )
        sign_likelihood_n = pm.Data(
            "sign_likelihood_n", signed_spec.trials, dims=("obs_sign_id",)
        )
        sign_is_conditional = pm.Data(
            "sign_is_conditional",
            signed_spec.is_conditional.astype(int),
            dims=("obs_sign_id",),
        )

        # Shared trend + HSGP builder (gp_utils); graph byte-identical to the
        # inlined form. Full-grid (n_all,) latents are returned as plain tensors
        # (not stored Deterministics): only the obs/plot/query slices below are
        # extracted, so storing the full arrays for every draw would waste a large
        # amount of trace memory.
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
            store_deterministic=False,
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
            store_deterministic=False,
            clamp_above_hi=_clamp_q,
        )

        # ---- Signed ratio: g_sign(a) -> r(a) = sigmoid(g_sign(a)) ----
        # Three-anchor "tent" mean (young / peak / old): r(a) is a developmental
        # hump, so the mean rises to the peak anchor then declines (clamped flat
        # outside), giving a hill-shaped prior median rather than the flat median an
        # intercept-only mean gives. The GP carries smooth departures.
        sa_young, sa_peak, sa_old = config.sign_anchor_ages
        g_sign_all = tent_and_gp(
            cfg_low=config.p_slope_low_sign_dist,
            cfg_mid=config.p_slope_mid_sign_dist,
            cfg_hi=config.p_slope_hi_sign_dist,
            z_low=(sa_young - X_obs_mean) / X_obs_std,
            z_mid=(sa_peak - X_obs_mean) / X_obs_std,
            z_hi=(sa_old - X_obs_mean) / X_obs_std,
            # Optional: estimate the peak age rather than assert it. Read from the
            # definition so no configuration class changes -- adding a field to a
            # definition class invalidates every existing fit of that class, and
            # nothing here should do that until the change is chosen deliberately.
            cfg_peak=(
                pz.Beta(
                    alpha=definition.sign_peak_prior[0],
                    beta=definition.sign_peak_prior[1],
                )
                if getattr(definition, "sign_peak_prior", None) is not None
                else None
            ),
            cfg_ell=config.ell_unit_sign_dist,
            cfg_eta=config.eta_sign_dist,
            suffix="_sign",
            X_all_z_data=X_all_z_data,
            grid=gp_grid,
            store_deterministic=False,
        )

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

        r_obs = pm.Deterministic("r_obs", r_all[i_obs0:i_obs1], dims=("obs_id",))
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
        # Kappa — signed
        # ============================================================

        kappa_sign_of_z = build_kappa_for_config(
            config, X_obs_mean=X_obs_mean, X_obs_std=X_obs_std, suffix="_sign"
        )

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
        #
        # Nested likelihoods: where a child outcome and understood are both
        # observed and logically nested, S | U and Sign | U use U trials with
        # means q and r. Rows without a usable U count retain marginal
        # likelihoods over the full inventory.

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

        # Signed likelihood (only where observed)
        p_sign_likelihood = pm.math.switch(
            sign_is_conditional,
            r_obs[idx_sign],
            p_sign_obs[idx_sign],
        )
        p_sign_likelihood = pm.math.clip(p_sign_likelihood, EPSILON, 1 - EPSILON)
        alpha_sign = p_sign_likelihood * kappa_sign_obs[idx_sign]
        beta_sign = (1 - p_sign_likelihood) * kappa_sign_obs[idx_sign]

        _ = pm.BetaBinomial(
            "y_sign_obs",
            n=sign_likelihood_n,
            alpha=alpha_sign,
            beta=beta_sign,
            observed=y_sign_observed,
            dims=("obs_sign_id",),
        )

    variables = pymc_utils.get_variables_dict(model_pm)

    pymc_utils.report_model_summary(model_pm)

    render_model_graph(model_pm, context.reporting.output_dir)

    context.set_model(model_pm, variables)


# ============================================================
# Sample extraction
# ============================================================


def extract_model_samples(trace: xr.DataTree) -> TrivariateModelSamples:
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

    # Signed rate
    g_sign_obs = _extract_posterior(trace, "g_sign_obs", "obs_id")
    g_sign_plot = _extract_posterior(trace, "g_sign_plot", "plot_id")
    g_sign_query = _extract_posterior(trace, "g_sign_query", "query_id")

    r_obs = _extract_posterior(trace, "r_obs", "obs_id")
    r_plot = _extract_posterior(trace, "r_plot", "plot_id")
    r_query = _extract_posterior(trace, "r_query", "query_id")

    # Signed (derived)
    f_sign_plot = _extract_posterior(trace, "f_sign_plot", "plot_id")
    f_sign_query = _extract_posterior(trace, "f_sign_query", "query_id")

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

    y_sign_obs_raw = np.array(trace.observed_data["y_sign_obs"].values, dtype=float)
    if int(obs_sign_mask.sum()) != y_sign_obs_raw.shape[0]:
        raise ValueError(
            f"obs_sign_mask count ({int(obs_sign_mask.sum())}) does not match observed "
            f"y_sign_obs length ({y_sign_obs_raw.shape[0]}); stored mask and likelihood "
            "rows are misaligned (issue #67)."
        )
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
    X_plot_z = _extract_posterior(trace, "z_plot", "plot_id")
    X_query_z = _extract_posterior(trace, "z_query", "query_id")

    return TrivariateModelSamples(
        X_obs=X_obs,
        X_plot=X_plot,
        X_query=X_query,
        X_plot_z=X_plot_z,
        X_query_z=X_query_z,
        f_u_plot=f_u_plot,
        f_u_query=f_u_query,
        p_u_plot=p_u_plot,
        p_u_query=p_u_query,
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
        f_sign_plot=f_sign_plot,
        f_sign_query=f_sign_query,
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


# ``sample`` is engine-agnostic (identical pm.sample() call in every engine) —
# reuse the shared implementation from common.py rather than redefining it.
sample = _shared_sample


def diagnostics(context: TrivariateContext):
    """Run diagnostics on the posterior samples.

    Thin wrapper over the shared engine (common.py): trivariate reports
    per-outcome LOO-CV for understood/spoken/signed. (It used to name the three
    observation-level kappas for the trace plot as well; an observation-sized
    variable never fitted under ArviZ's subplot cap, so they never rendered, and
    since 2026-08-23 the sampler does not store them.)
    """
    _shared_diagnostics(
        context,
        loo_var_names=(
            ("y_u_obs", "words understood"),
            ("y_s_obs", "words spoken"),
            ("y_sign_obs", "words signed"),
        ),
    )


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

    q_plot = context.model_variables["q_plot"]
    q_query = context.model_variables["q_query"]
    kappa_s_plot = context.model_variables["kappa_s_plot"]
    kappa_s_query = context.model_variables["kappa_s_query"]

    r_plot = context.model_variables["r_plot"]
    r_query = context.model_variables["r_query"]
    kappa_sign_plot = context.model_variables["kappa_sign_plot"]
    kappa_sign_query = context.model_variables["kappa_sign_query"]

    with context.model:
        # Understood — plot / query
        p_u_plot_clip = pm.math.clip(p_u_plot, EPSILON, 1 - EPSILON)
        y_u_plot = pm.BetaBinomial(
            "y_u_plot",
            n=n_trials,
            alpha=p_u_plot_clip * kappa_u_plot,
            beta=(1 - p_u_plot_clip) * kappa_u_plot,
            dims=("plot_id",),
        )
        p_u_query_clip = pm.math.clip(p_u_query, EPSILON, 1 - EPSILON)
        y_u_query = pm.BetaBinomial(
            "y_u_query",
            n=n_trials,
            alpha=p_u_query_clip * kappa_u_query,
            beta=(1 - p_u_query_clip) * kappa_u_query,
            dims=("query_id",),
        )
        # Spoken — plot / query
        q_plot_clip = pm.math.clip(q_plot, EPSILON, 1 - EPSILON)
        pm.BetaBinomial(
            "y_s_plot",
            n=y_u_plot,
            alpha=q_plot_clip * kappa_s_plot,
            beta=(1 - q_plot_clip) * kappa_s_plot,
            dims=("plot_id",),
        )
        q_query_clip = pm.math.clip(q_query, EPSILON, 1 - EPSILON)
        pm.BetaBinomial(
            "y_s_query",
            n=y_u_query,
            alpha=q_query_clip * kappa_s_query,
            beta=(1 - q_query_clip) * kappa_s_query,
            dims=("query_id",),
        )
        # Signed — plot / query
        r_plot_clip = pm.math.clip(r_plot, EPSILON, 1 - EPSILON)
        pm.BetaBinomial(
            "y_sign_plot",
            n=y_u_plot,
            alpha=r_plot_clip * kappa_sign_plot,
            beta=(1 - r_plot_clip) * kappa_sign_plot,
            dims=("plot_id",),
        )
        r_query_clip = pm.math.clip(r_query, EPSILON, 1 - EPSILON)
        pm.BetaBinomial(
            "y_sign_query",
            n=y_u_query,
            alpha=r_query_clip * kappa_sign_query,
            beta=(1 - r_query_clip) * kappa_sign_query,
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
            ("signed", "y_sign_obs", "obs_sign_mask"),
        ),
    )
    context.dataframes["posterior_predictive_calibration"] = calibration_df

    save_trace(trace, context.reporting.output_dir)

    sample_data = extract_model_samples(context.trace)
    context.set_model_samples(sample_data)


def _ratio_summary(X_query, ratio_query, ci_prob, interval_kind="eti"):
    """Median + inner-50%/outer credible interval for a ratio (q or r).

    Ratios are summarised with the reporting config's interval kind (ETI by
    default). Columns follow the neutral ``ci``/``ci50`` schema
    (:mod:`vocab_growth.intervals`).
    """
    return intervals.summarise(
        ratio_query, X_query, kind=interval_kind, outer=ci_prob, sample_axis=1
    )


def posterior_summary(context: TrivariateContext):
    """Compute and store the posterior summary tables at query ages."""
    samples = context.model_samples
    n_trials = context.model_data.n_trials
    ci_prob = context.reporting.ci_prob
    ci_kind = context.reporting.interval_kind
    # Per-quantity reporting caps, named by quantity rather than read off a cap
    # attribute (see vocab_growth.reporting_ages). r and p_any are ratios of
    # understood built from the signed ratio, so they take the tighter of the
    # comprehension and signing caps; until #238 they were trimmed with the
    # signing cap alone, which stopped being the tighter one when the
    # comprehension cap moved to 72 on 2026-08-22.
    config = context.model_config
    understood_cap = reporting_ages.max_age_for(
        config, reporting_ages.ReportedQuantity.UNDERSTOOD
    )
    signed_cap = reporting_ages.max_age_for(
        config, reporting_ages.ReportedQuantity.SIGNED
    )
    ratio_cap = reporting_ages.max_age_for(
        config, reporting_ages.ReportedQuantity.RATIO_OF_UNDERSTOOD
    )
    sign_ratio_cap = reporting_ages.max_age_for_sign_ratio(config)

    # Understood summary
    summary_u = posterior_analysis.posterior_summary_table(
        samples.X_query,
        samples.p_u_query,
        samples.y_u_query,
        n_trials=n_trials,
        ci_prob=ci_prob,
        interval_kind=ci_kind,
    )
    summary_u = posterior_analysis.trim_reported_ages(summary_u, understood_cap)
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
        interval_kind=ci_kind,
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
        ci_prob=ci_prob,
        interval_kind=ci_kind,
    )
    # Signed COUNTS stop where signing evidence stops (their own cap).
    summary_sign = posterior_analysis.trim_reported_ages(summary_sign, signed_cap)
    dataframe_table(
        summary_sign, title="Posterior summary — words signed", show_index=False
    )
    context.dataframes["posterior_summary_sign"] = summary_sign
    summary_sign.to_csv(
        os.path.join(context.reporting.output_dir, "posterior_summary_sign.csv"),
        index=False,
    )

    # Production rate q(a) summary
    summary_q = _ratio_summary(samples.X_query, samples.q_query, ci_prob, ci_kind)
    summary_q = summary_q.rename(
        columns={
            "median": "q_median",
            "ci50_lo": "q_ci50_lo",
            "ci50_hi": "q_ci50_hi",
            "ci_lo": "q_ci_lo",
            "ci_hi": "q_ci_hi",
        }
    )
    summary_q = posterior_analysis.trim_reported_ages(summary_q, ratio_cap)
    dataframe_table(
        summary_q, title="Posterior summary — production rate q(a)", show_index=False
    )
    context.dataframes["posterior_summary_q"] = summary_q
    summary_q.to_csv(
        os.path.join(context.reporting.output_dir, "posterior_summary_q.csv"),
        index=False,
    )

    # Signed rate r(a) summary
    summary_r = _ratio_summary(samples.X_query, samples.r_query, ci_prob, ci_kind)
    summary_r = summary_r.rename(
        columns={
            "median": "r_median",
            "ci50_lo": "r_ci50_lo",
            "ci50_hi": "r_ci50_hi",
            "ci_lo": "r_ci_lo",
            "ci_hi": "r_ci_hi",
        }
    )
    # r is a ratio of understood built from signing: tighter of the two caps.
    summary_r = posterior_analysis.trim_reported_ages(summary_r, sign_ratio_cap)
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
    inner = intervals.INNER_CI_PROB
    p_any_out = intervals.bands(samples.p_any_query, ci_prob, ci_kind, sample_axis=1)
    p_any_in = intervals.bands(samples.p_any_query, inner, ci_kind, sample_axis=1)
    Ey_any_out = intervals.bands(Ey_any, ci_prob, ci_kind, sample_axis=1)
    Ey_any_in = intervals.bands(Ey_any, inner, ci_kind, sample_axis=1)
    summary_p_any = pd.DataFrame(
        {
            "age_months": samples.X_query,
            "p_any_median": np.median(samples.p_any_query, axis=1),
            "p_any_ci50_lo": p_any_in[:, 0],
            "p_any_ci50_hi": p_any_in[:, 1],
            "p_any_ci_lo": p_any_out[:, 0],
            "p_any_ci_hi": p_any_out[:, 1],
            "Ey_any_median": np.median(Ey_any, axis=1),
            "Ey_any_ci50_lo": Ey_any_in[:, 0],
            "Ey_any_ci50_hi": Ey_any_in[:, 1],
            "Ey_any_ci_lo": Ey_any_out[:, 0],
            "Ey_any_ci_hi": Ey_any_out[:, 1],
        }
    )
    # p_any is p_U * (1 - (1 - r)(1 - q)) -- conditioned on understood and a
    # function of the signed ratio, so the tighter of those two caps binds
    # (see reporting_ages.max_age_for_sign_ratio).
    summary_p_any = posterior_analysis.trim_reported_ages(summary_p_any, sign_ratio_cap)
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
    max_age_months_understood: float | None = None,
    max_age_months_spoken: float | None = None,
    max_age_months_signed: float | None = None,
):
    """Plot understood, spoken and signed posterior predictive median trends.

    Three outcomes with three different evidence bases, so three caps: each
    series is trimmed independently and the companion CSV blanks each column
    past its own cap. See :mod:`vocab_growth.reporting_ages`.
    """
    X_plot = samples.X_plot
    _all = np.ones_like(X_plot, dtype=bool)
    ku = _all if max_age_months_understood is None else X_plot <= max_age_months_understood
    ks = _all if max_age_months_spoken is None else X_plot <= max_age_months_spoken
    ksg = _all if max_age_months_signed is None else X_plot <= max_age_months_signed

    outer, inner = intervals.DEFAULT_CI_PROB, intervals.INNER_CI_PROB
    y_u_median = np.quantile(samples.y_u_plot, 0.50, axis=1)
    y_u_ci = intervals.bands(samples.y_u_plot, outer, "eti", sample_axis=1)
    y_u_ci50 = intervals.bands(samples.y_u_plot, inner, "eti", sample_axis=1)

    y_s_median = np.quantile(samples.y_s_plot, 0.50, axis=1)
    y_s_ci = intervals.bands(samples.y_s_plot, outer, "eti", sample_axis=1)
    y_s_ci50 = intervals.bands(samples.y_s_plot, inner, "eti", sample_axis=1)

    y_sign_median = np.quantile(samples.y_sign_plot, 0.50, axis=1)
    y_sign_ci = intervals.bands(samples.y_sign_plot, outer, "eti", sample_axis=1)
    y_sign_ci50 = intervals.bands(samples.y_sign_plot, inner, "eti", sample_axis=1)

    fig, ax = plt.subplots(figsize=plot_styles.FIGSIZE_XL)

    ax.fill_between(X_plot[ku], y_u_ci[ku, 0], y_u_ci[ku, 1], alpha=0.12, color="C0")
    ax.fill_between(X_plot[ku], y_u_ci50[ku, 0], y_u_ci50[ku, 1], alpha=0.22, color="C0")
    ax.plot(X_plot[ku], y_u_median[ku], lw=3, color="C0", label="Words understood (median)")

    ax.fill_between(X_plot[ks], y_s_ci[ks, 0], y_s_ci[ks, 1], alpha=0.12, color="C1")
    ax.fill_between(X_plot[ks], y_s_ci50[ks, 0], y_s_ci50[ks, 1], alpha=0.22, color="C1")
    ax.plot(X_plot[ks], y_s_median[ks], lw=3, color="C1", label="Words spoken (median)")

    ax.fill_between(X_plot[ksg], y_sign_ci[ksg, 0], y_sign_ci[ksg, 1], alpha=0.12, color="C2")
    ax.fill_between(X_plot[ksg], y_sign_ci50[ksg, 0], y_sign_ci50[ksg, 1], alpha=0.22, color="C2")
    ax.plot(X_plot[ksg], y_sign_median[ksg], lw=3, color="C2", label="Words signed (median)")

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
        _blank_ku = np.where(ku, 1.0, np.nan)
        _blank_ks = np.where(ks, 1.0, np.nan)
        _blank_ksg = np.where(ksg, 1.0, np.nan)
        _save_csv(
            pd.DataFrame(
                {
                    "age_months": X_plot,
                    "understood_median": y_u_median * _blank_ku,
                    "understood_ci50_lo": y_u_ci50[:, 0] * _blank_ku,
                    "understood_ci50_hi": y_u_ci50[:, 1] * _blank_ku,
                    "understood_ci_lo": y_u_ci[:, 0] * _blank_ku,
                    "understood_ci_hi": y_u_ci[:, 1] * _blank_ku,
                    "spoken_median": y_s_median * _blank_ks,
                    "spoken_ci50_lo": y_s_ci50[:, 0] * _blank_ks,
                    "spoken_ci50_hi": y_s_ci50[:, 1] * _blank_ks,
                    "spoken_ci_lo": y_s_ci[:, 0] * _blank_ks,
                    "spoken_ci_hi": y_s_ci[:, 1] * _blank_ks,
                    "signed_median": y_sign_median * _blank_ksg,
                    "signed_ci50_lo": y_sign_ci50[:, 0] * _blank_ksg,
                    "signed_ci50_hi": y_sign_ci50[:, 1] * _blank_ksg,
                    "signed_ci_lo": y_sign_ci[:, 0] * _blank_ksg,
                    "signed_ci_hi": y_sign_ci[:, 1] * _blank_ksg,
                }
            )[ku | ks | ksg],
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
    label = "extrapolation (outside observed signing ages)"
    if x0 < lo:
        ax.axvspan(x0, lo, color="grey", alpha=0.12, lw=0, label=label)
        label = None
    if x1 > hi:
        ax.axvspan(hi, x1, color="grey", alpha=0.12, lw=0, label=label)
    ax.set_xlim(x0, x1)


def plot_signed_rate(
    samples: TrivariateModelSamples,
    ci_prob: float = intervals.DEFAULT_CI_PROB,
    interval_kind: intervals.IntervalKind = "eti",
    output_dir: str | None = None,
    filename: str | None = None,
    support_range: tuple[float, float] | None = None,
    max_age_months: float | None = None,
):
    """Plot the posterior of the signed ratio r(a) = p_Sign(a) / p_U(a) over age.

    ``support_range`` and ``max_age_months`` answer different questions and both
    apply. The shading marks where *signing* observations stop, disclosing rather
    than hiding the extrapolation; the cap stops the curve where *comprehension*
    reporting stops, because ``r`` is a ratio of comprehension and cannot be
    reported past the denominator's age range.
    """
    X_plot = samples.X_plot
    r_plot = samples.r_plot

    if max_age_months is not None:
        keep = np.asarray(X_plot) <= max_age_months
        X_plot = X_plot[keep]
        r_plot = r_plot[keep, :]

    r_median = np.median(r_plot, axis=1)
    r_ci = intervals.bands(r_plot, ci_prob, interval_kind, sample_axis=1)
    r_ci50 = intervals.bands(r_plot, intervals.INNER_CI_PROB, interval_kind, sample_axis=1)
    pct = int(round(ci_prob * 100))

    fig, ax = plt.subplots(figsize=plot_styles.FIGSIZE_XL)

    ax.fill_between(X_plot, r_ci[:, 0], r_ci[:, 1], alpha=0.20, label=f"{pct}% interval")
    ax.fill_between(X_plot, r_ci50[:, 0], r_ci50[:, 1], alpha=0.30, label="50% interval")
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
                    "ci50_lo": r_ci50[:, 0],
                    "ci50_hi": r_ci50[:, 1],
                    "ci_lo": r_ci[:, 0],
                    "ci_hi": r_ci[:, 1],
                }
            ),
            output_dir,
            filename,
        )

    return fig


def plot_sign_speech_crossover(
    samples: TrivariateModelSamples,
    ci_prob: float = intervals.DEFAULT_CI_PROB,
    interval_kind: intervals.IntervalKind = "eti",
    output_dir: str | None = None,
    filename: str | None = None,
    support_range: tuple[float, float] | None = None,
    max_age_months: float | None = None,
):
    """Plot signed rate r(a) against spoken rate q(a) — the sign->speech hand-off.

    Both ``q`` and ``r`` are ratios of comprehension, so ``max_age_months`` stops
    the pair where comprehension reporting stops; ``support_range`` separately
    shades where signing observations run out.
    """
    X_plot = samples.X_plot
    q_plot = samples.q_plot
    r_plot = samples.r_plot

    if max_age_months is not None:
        keep = np.asarray(X_plot) <= max_age_months
        X_plot = X_plot[keep]
        q_plot = q_plot[keep, :]
        r_plot = r_plot[keep, :]

    q_median = np.median(q_plot, axis=1)
    q_hdi = intervals.bands(q_plot, ci_prob, interval_kind, sample_axis=1)
    r_median = np.median(r_plot, axis=1)
    r_hdi = intervals.bands(r_plot, ci_prob, interval_kind, sample_axis=1)

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
                    "q_ci_lo": q_hdi[:, 0],
                    "q_ci_hi": q_hdi[:, 1],
                    "r_median": r_median,
                    "r_ci_lo": r_hdi[:, 0],
                    "r_ci_hi": r_hdi[:, 1],
                }
            ),
            output_dir,
            filename,
        )

    return fig


def _multi_outcome_frame(
    ages: np.ndarray,
    series: dict[str, tuple[np.ndarray, float | None]],
) -> pd.DataFrame:
    """One age column shared by several series that stop at different ages.

    The convention is ``joint_trajectory``'s, which had it first: the age column
    runs to the **widest** of the caps, and every series is NaN past its own. So
    the file never carries a row no series reports on, and never implies a series
    was reported where it was not.

    Trimming each series to its own length instead is what broke
    ``plot_modality_trajectories``: the frame then has columns of three different
    lengths and pandas refuses to build it at all. That surfaced only when VG14
    was first refitted after per-outcome caps arrived, because the figure is
    written before the CSV and was always correct.
    """
    caps = [cap for _, cap in series.values() if cap is not None]
    keep = np.ones_like(ages, dtype=bool) if len(caps) < len(series) else ages <= max(caps)
    data: dict[str, np.ndarray] = {"age_months": ages[keep]}
    for name, (values, cap) in series.items():
        trimmed = np.asarray(values, dtype=float)[keep]
        data[name] = (
            trimmed if cap is None else np.where(ages[keep] <= cap, trimmed, np.nan)
        )
    return pd.DataFrame(data)


def plot_modality_trajectories(
    samples: TrivariateModelSamples,
    n_trials: int,
    ci_prob: float = intervals.DEFAULT_CI_PROB,
    interval_kind: intervals.IntervalKind = "eti",
    output_dir: str | None = None,
    filename: str | None = None,
    max_age_months_understood: float | None = None,
    max_age_months_spoken: float | None = None,
    max_age_months_signed: float | None = None,
    max_age_months_any: float | None = None,
):
    """Plot expected p_U, p_S, p_Sign and p_any trajectories (in word counts).

    Each curve stops at its own outcome's reporting cap, as the joint-trajectory
    figure beside it already did. Without the caps this figure ran the full plot
    grid -- for VG14 that is 115 months, thirty past the comprehension and
    signing caps -- directly above a ``posterior_summary_p_any`` table trimmed
    to 84. The policy test could not see it: ``modality_trajectories`` has no
    outcome suffix, so it matched no entry in the test's stem map.

    ``p_any`` takes its own explicit cap, computed at the call site by
    :func:`vocab_growth.reporting_ages.max_age_for_sign_ratio`: it is a ratio of
    understood built from the signed ratio, so the tighter of the comprehension
    and signing caps binds. This function used to derive the cap itself as
    ``min(spoken, signed)`` -- the components rule alone -- which stopped being
    the tighter rule when the comprehension cap moved to 72 on 2026-08-22 (#238).
    """
    X_plot = samples.X_plot

    def _trim(values: np.ndarray, cap: float | None) -> tuple[np.ndarray, np.ndarray]:
        if cap is None:
            return X_plot, values
        keep = X_plot <= cap
        return X_plot[keep], values[keep]

    def _masked(values: np.ndarray, cap: float | None) -> np.ndarray:
        """The same trim, kept on the full grid with NaN past the cap.

        The figure wants each curve against its own shortened x; the CSV wants
        one age column shared by every series. Trimming for both is what broke
        this function: the CSV paired the FULL ``X_plot`` with arrays cut at
        three different caps, so ``pd.DataFrame`` raised "All arrays must be of
        the same length" the first time a trivariate model was refitted after
        per-outcome caps arrived. Nothing caught it earlier because the figure
        is written before the CSV -- the plot was always correct -- and because
        ``modality_trajectories`` carries no outcome suffix, so the reporting-age
        policy test matches no entry for it (see this function's docstring).
        """
        if cap is None:
            return values
        return np.where(X_plot <= cap, values, np.nan)

    any_cap = max_age_months_any

    E_u_full = np.median(samples.p_u_plot, axis=1) * n_trials
    E_s_full = np.median(samples.p_s_plot, axis=1) * n_trials
    E_sign_full = np.median(samples.p_sign_plot, axis=1) * n_trials
    E_any_full = np.median(samples.p_any_plot, axis=1) * n_trials
    any_hdi_full = intervals.bands(
        samples.p_any_plot * n_trials, ci_prob, interval_kind, sample_axis=1
    )

    x_u, E_u = _trim(E_u_full, max_age_months_understood)
    x_s, E_s = _trim(E_s_full, max_age_months_spoken)
    x_sign, E_sign = _trim(E_sign_full, max_age_months_signed)
    x_any, E_any = _trim(E_any_full, any_cap)
    _, any_hdi = _trim(any_hdi_full, any_cap)

    fig, ax = plt.subplots(figsize=plot_styles.FIGSIZE_XL)

    ax.fill_between(x_any, any_hdi[:, 0], any_hdi[:, 1], alpha=0.12, color="C3")
    ax.plot(x_any, E_any, lw=3, color="C3", label="Total expressive p_any")
    ax.plot(x_u, E_u, lw=2.5, color="C0", label="Understood p_U")
    ax.plot(x_s, E_s, lw=2.5, color="C1", label="Spoken p_S")
    ax.plot(x_sign, E_sign, lw=2.5, color="C2", label="Signed p_Sign")

    ax.set_xlabel("Age (months)")
    ax.set_ylabel("Expected word count")
    ax.set_ylim(-20, n_trials + 50)
    ax.legend(loc="upper left", frameon=True)
    ax.set_title("Expected vocabulary by modality")

    if output_dir is not None and filename is not None:
        fig.savefig(os.path.join(output_dir, f"{filename}.png"), dpi=300)
        fig.savefig(os.path.join(output_dir, f"{filename}.svg"))
        _save_csv(
            _multi_outcome_frame(
                X_plot,
                {
                    "understood_median": (E_u_full, max_age_months_understood),
                    "spoken_median": (E_s_full, max_age_months_spoken),
                    "signed_median": (E_sign_full, max_age_months_signed),
                    "any_median": (E_any_full, any_cap),
                    "any_ci_lo": (any_hdi_full[:, 0], any_cap),
                    "any_ci_hi": (any_hdi_full[:, 1], any_cap),
                },
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
):
    """Run the standard per-outcome plotting pipeline for a trivariate outcome."""
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


def plot_p_any_validation(
    samples: TrivariateModelSamples,
    output_dir: str | None = None,
    filename: str | None = None,
    window: tuple[float, float] = (20.0, 56.0),
):
    """Validate the independence-based p_any against uk_02's observed union.

    p_any assumes sign and speech are conditionally independent given age. uk_02
    was the first source with the four-cell breakdown (sign-only / sign+speech /
    speech-only / understood-only), so we can compute the *observed* fraction of
    understood words produced in any modality and compare it with the model's
    union p_any / p_U over the overlap window. The model union systematically
    exceeds the observed union: the sign-speech association is positive, so
    independence over-states the total.

    This is a **uk_02-specific check, not a general validation** of conditional
    independence: uk_07, es_01 and nz_01 also carry cross-tabulations, with
    materially different descriptive associations by source (#238). VG15
    identifies the association from all four and is the model of record for it.

    The model side of ``model_gap_pp`` is evaluated per posterior draw **at
    uk_02's observed ages** and averaged over those same rows, so observed and
    modelled unions share one empirical age distribution and the gap carries a
    posterior interval. It previously averaged the pointwise median over an
    equally spaced grid, so part of any reported gap was age weighting rather
    than model behaviour, and no interval was supplied (#238).
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
    union_ci = intervals.bands(union_draws, intervals.DEFAULT_CI_PROB, "eti", sample_axis=1)
    union_pct = int(round(intervals.DEFAULT_CI_PROB * 100))

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
        Xw, union_ci[:, 0], union_ci[:, 1], alpha=0.20, color="C3",
        label=f"VG14 p_any / p_U ({union_pct}% interval, independence)",
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
    # Pure independence bias, computed within uk_02 (its own r,q): isolates the
    # positive sign-speech association from model fit.
    indep_bias_pp = 100.0 * (indep_mean - obs_mean)
    # Model check: VG14's independence union vs the observed union (conflates the
    # association with the model's r,q differing from uk_02's in-sample r,q).
    # Per draw, at the observed ages, so the two sides share one age
    # distribution and the gap has an interval -- see the docstring.
    obs_ages = raw["age"].to_numpy(dtype=float)
    union_at_obs = np.empty((len(obs_ages), union_draws.shape[1]))
    for d in range(union_draws.shape[1]):
        union_at_obs[:, d] = np.interp(obs_ages, Xw, union_draws[:, d])
    model_mean_draws = union_at_obs.mean(axis=0)
    gap_draws = 100.0 * (model_mean_draws - obs_mean)
    gap_lo, gap_hi = intervals.interval_1d(gap_draws, intervals.DEFAULT_CI_PROB, "eti")
    mod_mean = float(np.median(model_mean_draws))
    model_gap_pp = float(np.median(gap_draws))

    if output_dir is not None and filename is not None:
        fig.savefig(os.path.join(output_dir, f"{filename}.png"), dpi=300)
        fig.savefig(os.path.join(output_dir, f"{filename}.svg"))
        _save_csv(
            pd.DataFrame(
                {
                    "age_months": Xw,
                    "model_union_median": union_med,
                    "model_union_ci_lo": union_ci[:, 0],
                    "model_union_ci_hi": union_ci[:, 1],
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
                    "model_gap_pp_ci_lo": [gap_lo],
                    "model_gap_pp_ci_hi": [gap_hi],
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
            ("VG14 p_any / p_U at observed ages (median)", round(mod_mean, 3)),
            (
                "VG14 vs observed union (pp, median [89% ETI])",
                f"{model_gap_pp:.1f} [{gap_lo:.1f}, {gap_hi:.1f}]",
            ),
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
    sign_ages = analysis_df.loc[has_sign, "age"].dropna()
    signing_support_range = (
        (float(sign_ages.min()), float(sign_ages.max()))
        if not sign_ages.empty
        else None
    )
    n_trials = context.model_data.n_trials
    ci_prob = context.reporting.ci_prob
    output_dir = context.reporting.output_dir

    # Per-quantity reporting caps (vocab_growth.reporting_ages). Named locals,
    # one per quantity, so each call below visibly carries the cap for the
    # quantity it draws -- tests/test_reporting_age_caps.py asserts exactly this
    # by AST. r(a), the crossover and p_any are ratios of understood built from
    # the signed ratio, so they take the tighter of the comprehension and
    # signing caps; until #238 they carried the signing cap alone.
    config = context.model_config
    understood_cap = reporting_ages.max_age_for(
        config, reporting_ages.ReportedQuantity.UNDERSTOOD
    )
    spoken_cap = reporting_ages.max_age_for(
        config, reporting_ages.ReportedQuantity.SPOKEN
    )
    signed_cap = reporting_ages.max_age_for(
        config, reporting_ages.ReportedQuantity.SIGNED
    )
    ratio_cap = reporting_ages.max_age_for(
        config, reporting_ages.ReportedQuantity.RATIO_OF_UNDERSTOOD
    )
    sign_ratio_cap = reporting_ages.max_age_for_sign_ratio(config)

    # ---- Joint trajectory (understood, spoken, signed) ----
    fig = plot_understood_spoken_signed_trajectory(
        samples,
        n_trials=n_trials,
        output_dir=output_dir,
        filename="joint_trajectory",
        max_age_months_understood=understood_cap,
        max_age_months_spoken=spoken_cap,
        max_age_months_signed=signed_cap,
    )
    context.plots["joint_trajectory"] = fig
    plt.close(fig)

    # ---- Modality trajectories (p_U, p_S, p_Sign, p_any) ----
    fig = plot_modality_trajectories(
        samples,
        n_trials=n_trials,
        ci_prob=ci_prob,
        output_dir=output_dir,
        filename="modality_trajectories",
        max_age_months_understood=understood_cap,
        max_age_months_spoken=spoken_cap,
        max_age_months_signed=signed_cap,
        max_age_months_any=sign_ratio_cap,
    )
    context.plots["modality_trajectories"] = fig
    plt.close(fig)

    # ---- Production rate q(a) ----
    fig = plot_production_rate(
        samples,
        ci_prob=ci_prob,
        output_dir=output_dir,
        filename="production_rate",
        max_age_months=ratio_cap,
    )
    context.plots["production_rate"] = fig
    plt.close(fig)

    # ---- Signed rate r(a) (extrapolation outside signing support shaded) ----
    fig = plot_signed_rate(
        samples,
        ci_prob=ci_prob,
        output_dir=output_dir,
        filename="signed_rate",
        support_range=signing_support_range,
        max_age_months=sign_ratio_cap,
    )
    context.plots["signed_rate"] = fig
    plt.close(fig)

    # ---- Sign -> speech crossover ----
    fig = plot_sign_speech_crossover(
        samples,
        ci_prob=ci_prob,
        output_dir=output_dir,
        filename="sign_speech_crossover",
        support_range=signing_support_range,
        max_age_months=sign_ratio_cap,
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
        ci_prob=ci_prob,
        output_dir=output_dir,
        filename="comprehension_production_gap",
        max_age_months=ratio_cap,
    )
    context.plots["comprehension_production_gap"] = fig
    plt.close(fig)

    # ---- Per-outcome plots: understood ----
    _run_trivariate_outcome_plots(
        samples=samples,
        y_plot=samples.y_u_plot,
        y_query=samples.y_u_query,
        f_plot=samples.f_u_plot,
        p_plot=samples.p_u_plot,
        kappa_plot=samples.kappa_u_plot,
        kappa_query=samples.kappa_u_query,
        x_obs=analysis_df.loc[has_u, "age"],
        y_obs=analysis_df.loc[has_u, "understood"],
        n_trials=n_trials,
        ci_prob=ci_prob,
        output_dir=output_dir,
        suffix="u",
        outcome_label="Words understood",
        y_label="Predicted words understood",
        max_age_months=understood_cap,
    )

    # ---- Per-outcome plots: spoken ----
    _run_trivariate_outcome_plots(
        samples=samples,
        y_plot=samples.y_s_plot,
        y_query=samples.y_s_query,
        f_plot=samples.f_s_plot,
        p_plot=samples.p_s_plot,
        kappa_plot=samples.kappa_s_plot,
        kappa_query=samples.kappa_s_query,
        x_obs=analysis_df.loc[has_s, "age"],
        y_obs=analysis_df.loc[has_s, "spoken"],
        n_trials=n_trials,
        ci_prob=ci_prob,
        output_dir=output_dir,
        suffix="s",
        outcome_label="Words spoken",
        y_label="Predicted words spoken",
        max_age_months=spoken_cap,
    )

    # ---- Per-outcome plots: signed ----
    _run_trivariate_outcome_plots(
        samples=samples,
        y_plot=samples.y_sign_plot,
        y_query=samples.y_sign_query,
        f_plot=samples.f_sign_plot,
        p_plot=samples.p_sign_plot,
        kappa_plot=samples.kappa_sign_plot,
        kappa_query=samples.kappa_sign_query,
        x_obs=analysis_df.loc[has_sign, "age"],
        y_obs=analysis_df.loc[has_sign, "signed"],
        n_trials=n_trials,
        ci_prob=ci_prob,
        output_dir=output_dir,
        suffix="sign",
        outcome_label="Words signed",
        y_label="Predicted words signed",
        max_age_months=signed_cap,
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
    return run_fit_pipeline(
        config,
        definition,
        stages=[
            ("Prepare data", lambda ctx: prepare_trivariate_data(ctx, definition)),
            (
                "Priors and hyperparameters",
                lambda ctx: configure_trivariate_priors(ctx, definition),
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
            ("Plots", _run_trivariate_plots),
            ("Report", report),
        ],
    )
