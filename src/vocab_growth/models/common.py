# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Shared dataclasses and pipeline functions for the vocabulary growth model family.
"""

import os
import shutil
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Generic, TypeVar

import arviz as az
import dse_research_utils.environment.info as env_info
import dse_research_utils.math.constants as math_constants
import dse_research_utils.metadata.packages as package_metadata
import dse_research_utils.plot.diagnostics_mcmc as plot_diagnostics_mcmc
import dse_research_utils.plot.distributions as plot_dist
import dse_research_utils.plot.predictive as plot_predictive
import dse_research_utils.plot.styles as plot_styles
import dse_research_utils.statistics.descriptive as descriptive_stats
import dse_research_utils.statistics.diagnostics as shared_diagnostics
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
from arviz import ELPDData
from matplotlib.figure import Figure
from preliz.distributions.distributions import Continuous

import vocab_growth.data_utils as vocab_data_utils
import vocab_growth.environment as local_env
import vocab_growth.plotting as plotting
import vocab_growth.posterior_analysis as posterior_analysis
import vocab_growth.reporting as vg_reporting
from vocab_growth.models.build_utils import (
    construct_age_grids,
    slope_anchor_logit_coeffs,
    standardize_ages,
    validate_ell_bounds,
)
from vocab_growth.models.definitions import UnivariateModelDefinition
from vocab_growth.models.diagnostics_utils import capped_plot_var_names
from vocab_growth.models.gp_utils import GPGrid, build_kappa_of_z, trend_and_gp
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


@dataclass
class BaseModelConfiguration:
    """Base configuration shared by all model variants."""

    slope_anchors: tuple[float, float]
    """Reference ages (months) for the slope parameterisation."""
    ell_months_range: tuple[int, int]
    """Range of length-scales in months for the HSGP prior."""
    n_plot: int
    """Number of points for plotting the developmental trajectory."""
    ages_query: list[int]
    """Ages in months for querying the model."""

    def __post_init__(self) -> None:
        if len(self.slope_anchors) != 2:
            raise ValueError("slope_anchors must be a tuple of two float values.")
        if len(self.ell_months_range) != 2:
            raise ValueError("ell_months_range must be a tuple of two int values.")
        if self.n_plot <= 0:
            raise ValueError("n_plot must be a positive integer.")
        if not isinstance(self.ages_query, list) or len(self.ages_query) == 0:
            raise ValueError("ages_query must be a non-empty list of integers.")


@dataclass
class ModelConfiguration(BaseModelConfiguration):
    """Configuration for a single-outcome model (VG01-VG04)."""

    p_slope_low_dist: Continuous
    """Prior distribution for the mean proportion of the outcome at the lower slope anchor."""
    p_slope_hi_dist: Continuous
    """Prior distribution for the mean proportion of the outcome at the upper slope anchor."""
    ell_unit_dist: Continuous
    """Prior distribution for the unit length-scale parameter (ell_unit)."""
    eta_dist: Continuous
    """Prior distribution for the GP amplitude (eta)."""
    kappa_min_dist: Continuous
    """Prior distribution for the minimum kappa value (kappa_min)."""
    a_kappa_dist: Continuous
    """Prior distribution for the age slope of kappa (a_kappa)."""
    b_kappa_mag_dist: Continuous
    """Prior distribution for the magnitude of the kappa parameter."""


@dataclass
class ModelSamples:
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
    f_obs: np.ndarray
    """Posterior samples of the latent linear predictor f for the observed points: (n, n_samples)"""
    f_plot: np.ndarray
    """Posterior samples of the latent linear predictor f for the plot points: (n_plot, n_samples)"""
    f_query: np.ndarray
    """Posterior samples of the latent linear predictor f for the query points: (n_query, n_samples)"""
    p_obs: np.ndarray
    """Posterior samples of p(z) for the observed points: (n, n_samples)"""
    p_plot: np.ndarray
    """Posterior samples of p(z) for the plot points: (n_plot, n_samples)"""
    p_query: np.ndarray
    """Posterior samples of p(z) for the query points: (n_query, n_samples)"""
    y_obs: np.ndarray
    """Observed vocabulary sizes, shape (n,)."""
    y_plot: np.ndarray
    """Posterior predictive samples of y for the plot points: (n_plot, n_samples)"""
    y_query: np.ndarray
    """Posterior predictive samples of y for the query points: (n_query, n_samples)"""
    kappa_plot: np.ndarray
    """Posterior samples of kappa for the plot points: (n_plot, n_samples)"""
    kappa_query: np.ndarray
    """Posterior samples of kappa for the query points: (n_query, n_samples)"""


C = TypeVar("C", bound=BaseModelConfiguration, default=ModelConfiguration)
S = TypeVar("S", default=ModelSamples)


@dataclass
class ModelFitContext(Generic[C, S]):
    reporting: reporting.ReportingConfiguration
    sampling: sampling.SamplingConfiguration
    execution_context: str = field(default_factory=env_info.get_execution_context)
    plots: dict[str, Figure] = field(default_factory=dict)
    dataframes: dict[str, pd.DataFrame] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)
    _model_data: model_data.BinomialModelData | None = None
    _analysis_df: pd.DataFrame | None = None
    _model_config: C | None = None
    _model: pm.Model | None = None
    _model_variables: dict | None = None
    _prior_samples: xr.DataTree | None = None
    _trace: xr.DataTree | None = None
    _loocv: ELPDData | dict[str, ELPDData] | None = None
    _model_samples: S | None = None

    def is_script(self) -> bool:
        """Check if the execution context is a script."""
        return self.execution_context == "script"

    def is_interactive(self) -> bool:
        """Check if the execution context is interactive (e.g., Jupyter notebook)."""
        return self.execution_context in ("jupyter", "ipython", "interactive-other")

    def set_model_data(
        self, model_data: model_data.BinomialModelData, analysis_df: pd.DataFrame
    ):
        self._model_data = model_data
        self._analysis_df = analysis_df

    @property
    def model_data(self) -> model_data.BinomialModelData:
        if self._model_data is None:
            raise ValueError("Model data has not been set in the context.")
        return self._model_data

    @property
    def analysis_df(self) -> pd.DataFrame:
        if self._analysis_df is None:
            raise ValueError("Analysis DataFrame has not been set in the context.")
        return self._analysis_df

    def set_model_config(self, model_config: C):
        self._model_config = model_config

    @property
    def model_config(self) -> C:
        if self._model_config is None:
            raise ValueError("Model configuration has not been set in the context.")
        return self._model_config

    def set_model(self, model: pm.Model, variables: dict):
        self._model = model
        self._model_variables = variables

    @property
    def model(self) -> pm.Model:
        if self._model is None:
            raise ValueError("Model has not been set in the context.")
        return self._model

    @property
    def model_variables(self) -> dict:
        if self._model_variables is None:
            raise ValueError("Model variables have not been set in the context.")
        return self._model_variables

    def set_prior_samples(self, prior_samples: xr.DataTree):
        self._prior_samples = prior_samples

    @property
    def prior_samples(self) -> xr.DataTree:
        if self._prior_samples is None:
            raise ValueError("Prior samples have not been set in the context.")
        return self._prior_samples

    def set_trace(self, trace: xr.DataTree):
        self._trace = trace

    @property
    def trace(self) -> xr.DataTree:
        if self._trace is None:
            raise ValueError("Trace has not been set in the context.")
        return self._trace

    def set_loocv(self, loocv: ELPDData | dict[str, ELPDData]):
        self._loocv = loocv

    @property
    def loocv(self) -> ELPDData | dict[str, ELPDData]:
        if self._loocv is None:
            raise ValueError("LOO-CV data has not been set in the context.")
        return self._loocv

    def set_model_samples(self, model_samples: S):
        self._model_samples = model_samples

    @property
    def model_samples(self) -> S:
        if self._model_samples is None:
            raise ValueError("Model samples have not been set in the context.")
        return self._model_samples


# ============================================================
# Shared pipeline functions for single-outcome models (VG01-VG04)
# ============================================================


PACKAGE_LIST = [
    "arviz",
    "matplotlib",
    "numba",
    "numpy",
    "numpyro",
    "pandas",
    "pymc",
    "pytensor",
]


def get_hsgp_hyperparams(
    X_all_z,
    ell_range_z,
):
    """
    Compute the HSGP boundary ``L`` and basis size ``M`` for the age kernel.

    Parameters
    ----------
    X_all_z : np.ndarray
        Standardised ages for the full evaluation grid the HSGP basis is
        evaluated on (observed + plot + query + anchor points), shape (n, 1).
        Passing only the observed ages under-covers the domain whenever the
        query grid extends beyond the observed range.
    ell_range_z : tuple of float
        Length-scale range in z-score scale.

    Returns
    -------
    tuple[list[float], list[int]]
        ``(L, M)`` where ``L`` is the HSGP boundary and ``M`` is the basis
        size, each wrapped in a single-element list (one per input dim).
    """
    x_min = float(np.min(X_all_z))
    x_max = float(np.max(X_all_z))

    ell_low_z = ell_range_z[0]
    ell_high_z = ell_range_z[1]

    m, c = pm.gp.hsgp_approx.approx_hsgp_hyperparams(
        x_range=[x_min, x_max],
        lengthscale_range=[ell_low_z, ell_high_z],
        cov_func="expquad",
    )

    # HSGP.prior_linearized centres X at the grid midpoint, so the domain the
    # basis must cover is the half-range S = (x_max - x_min) / 2 — the same S
    # that approx_hsgp_hyperparams sizes (m, c) for. Deriving L from max|z|
    # instead would inflate L past what m supports (z-scores are skewed about
    # zero, not about the midpoint), raising the smallest well-approximated
    # length-scale above ell_range_z[0].
    S = (x_max - x_min) / 2.0
    L = [S * c]
    M = [m]

    return L, M


def render_model_graph(model: pm.Model, output_dir: str) -> None:
    """Render the model DAG to ``gp_model_graph.svg`` in ``output_dir``.

    Best-effort: if the graphviz ``dot`` executable is not installed the render
    is skipped with a warning rather than aborting the fit, so the pipeline runs
    on machines without graphviz. The model-diagram figure is a non-essential
    reporting artefact (a missing SVG only shows as a broken figure in the
    optional Quarto report).
    """
    try:
        digraph = pymc_utils.model_to_graphviz(model)
        digraph.render(
            filename=os.path.join(output_dir, "gp_model_graph"),
            format="svg",
            cleanup=True,
        )
    except Exception as exc:  # e.g. graphviz 'dot' not on PATH — non-fatal
        console.print(f"[yellow]Skipped model graph: {exc}[/yellow]")


def extract_model_samples(trace: xr.DataTree) -> ModelSamples:
    """
    Extract model samples into a structured format for plotting and reporting.
    """

    # Posterior samples of the latent linear predictor f: (n, n_samples)
    f_obs = np.array(
        trace.posterior["f_obs"]
        .stack(sample=("chain", "draw"))
        .transpose("obs_id", "sample")
        .values
    )

    # Posterior samples of the latent linear predictor f for the plot points: (n_plot, n_samples)
    f_plot = np.array(
        trace.posterior["f_plot"]
        .stack(sample=("chain", "draw"))
        .transpose("plot_id", "sample")
        .values
    )

    # Posterior samples of the latent linear predictor f for the query points: (n_query, n_samples)
    f_query = np.array(
        trace.posterior["f_query"]
        .stack(sample=("chain", "draw"))
        .transpose("query_id", "sample")
        .values
    )

    p_obs = np.array(
        trace.posterior["p_obs"]
        .stack(sample=("chain", "draw"))
        .transpose("obs_id", "sample")
        .values
    )

    # Posterior samples of p(z) for the plot points: (n_plot, n_samples)
    p_plot = np.array(
        trace.posterior["p_plot"]
        .stack(sample=("chain", "draw"))
        .transpose("plot_id", "sample")
        .values
    )

    # Posterior samples of p(z) for the query points: (n_query, n_samples)
    p_query = np.array(
        trace.posterior["p_query"]
        .stack(sample=("chain", "draw"))
        .transpose("query_id", "sample")
        .values
    )

    # Posterior samples of kappa for the plot points: (n_plot, n_samples)
    kappa_plot = np.array(
        trace.posterior["kappa_plot"]
        .stack(sample=("chain", "draw"))
        .transpose("plot_id", "sample")
        .values
    )

    # Posterior samples of kappa for the query points: (n_query, n_samples)
    kappa_query = np.array(
        trace.posterior["kappa_query"]
        .stack(sample=("chain", "draw"))
        .transpose("query_id", "sample")
        .values
    )

    y_obs = np.array(trace.observed_data["y_obs"].values, dtype=int)

    # Posterior predictive samples of Y for the plot points: (n_plot, n_samples)
    y_plot = np.array(
        trace.posterior_predictive["y_plot"]
        .stack(sample=("chain", "draw"))
        .transpose("plot_id", "sample")
        .values,
        dtype=int,
    )

    # Posterior predictive samples of Y for the query points: (n_query, n_samples)
    y_query = np.array(
        trace.posterior_predictive["y_query"]
        .stack(sample=("chain", "draw"))
        .transpose("query_id", "sample")
        .values,
        dtype=int,
    )

    X_obs = np.array(trace.constant_data["X_obs"].values)

    X_plot = np.array(trace.constant_data["X_plot"].values)

    X_query = np.array(trace.constant_data["X_query"].values)

    X_obs_z = np.array(
        trace.posterior["z_obs"]
        .stack(sample=("chain", "draw"))
        .transpose("obs_id", "sample")
        .values
    )

    X_plot_z = np.array(
        trace.posterior["z_plot"]
        .stack(sample=("chain", "draw"))
        .transpose("plot_id", "sample")
        .values
    )

    X_query_z = np.array(
        trace.posterior["z_query"]
        .stack(sample=("chain", "draw"))
        .transpose("query_id", "sample")
        .values
    )

    model_samples = ModelSamples(
        X_obs=X_obs,
        X_plot=X_plot,
        X_query=X_query,
        X_obs_z=X_obs_z,
        X_plot_z=X_plot_z,
        X_query_z=X_query_z,
        f_obs=f_obs,
        f_plot=f_plot,
        f_query=f_query,
        p_obs=p_obs,
        p_plot=p_plot,
        p_query=p_query,
        y_obs=y_obs,
        y_plot=y_plot,
        y_query=y_query,
        kappa_plot=kappa_plot,
        kappa_query=kappa_query,
    )

    return model_samples


def build_model(context: ModelFitContext):
    """
    Builds vocabulary growth model.
    """
    n = len(context.model_data.y_obs)

    if context.model_data.X_obs.shape[0] != n:
        raise ValueError("X_obs and y_obs have inconsistent lengths.")
    if not np.all(0 <= context.model_data.y_obs):
        raise ValueError("y_obs contains negative counts.")
    if not np.all(context.model_data.y_obs <= context.model_data.n_trials):
        raise ValueError("y_obs exceeds n_trials.")

    X_obs_median = float(np.median(context.model_data.X_obs))
    X_obs_mean, X_obs_std, X_obs_z = standardize_ages(context.model_data.X_obs)

    key_value_table(
        "Build configuration",
        [
            ("Number of observations", n),
            ("Number of trials (n_trials)", context.model_data.n_trials),
            ("Slope anchors (months)", context.model_config.slope_anchors),
            ("Length-scale range (months)", context.model_config.ell_months_range),
            ("Number of plot points", context.model_config.n_plot),
            ("Query ages (months)", context.model_config.ages_query),
            ("Age median (months)", X_obs_median),
            ("Age mean (months)", X_obs_mean),
            ("Age std (months)", X_obs_std),
        ],
    )

    key_value_table(
        "Priors",
        [
            ("p_slope_low", context.model_config.p_slope_low_dist),
            ("p_slope_hi", context.model_config.p_slope_hi_dist),
            ("ell_unit", context.model_config.ell_unit_dist),
            ("eta", context.model_config.eta_dist),
            ("kappa_min", context.model_config.kappa_min_dist),
            ("a_kappa", context.model_config.a_kappa_dist),
            ("b_kappa_mag", context.model_config.b_kappa_mag_dist),
        ],
        key_header="Parameter",
        value_header="Distribution",
    )

    # Plot / query grids (standardised), stacked for 'free' predictions — see
    # models.build_utils.construct_age_grids.
    grids = construct_age_grids(
        context.model_data.X_obs,
        X_obs_z,
        X_obs_mean=X_obs_mean,
        X_obs_std=X_obs_std,
        n_plot=context.model_config.n_plot,
        ages_query=context.model_config.ages_query,
        slope_anchors=context.model_config.slope_anchors,
    )
    X_plot = grids.X_plot
    X_plot_z = grids.X_plot_z
    X_query = grids.X_query
    X_query_z = grids.X_query_z
    X_all_z = grids.X_all_z
    n_plot = grids.n_plot
    n_query = grids.n_query
    n_all = grids.n_all

    ell_low_months, ell_high_months = validate_ell_bounds(
        context.model_config.ell_months_range
    )
    ell_low_z = ell_low_months / X_obs_std
    ell_high_z = ell_high_months / X_obs_std
    ell_range_z = (ell_low_z, ell_high_z)

    L, M = get_hsgp_hyperparams(
        X_all_z,
        ell_range_z,
    )

    slope_age_a_z, slope_age_b_z = slope_anchor_logit_coeffs(
        context.model_config.slope_anchors,
        X_obs_mean=X_obs_mean,
        X_obs_std=X_obs_std,
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

    i_obs0, i_obs1 = 0, X_obs_z.shape[0]
    i_plot0, i_plot1 = i_obs1, i_obs1 + X_plot_z.shape[0]
    i_query0, i_query1 = i_plot1, i_plot1 + X_query_z.shape[0]

    coords = {
        "all_id": np.arange(n_all),
        "obs_id": np.arange(n),
        "plot_id": np.arange(n_plot),
        "query_id": np.arange(n_query),
        "x_dim": np.arange(1),
    }

    with pm.Model(coords=coords) as model:

        # ---- Data ----

        # Age data (standardised) for all points (obs + plot + query)
        X_all_z_data = pm.Data("X_all_z", X_all_z, dims=("all_id", "x_dim"))

        # Ages (in months) for plotting/reporting (observed, predictive plots and specific query points)
        _ = pm.Data("X_obs", context.model_data.X_obs.flatten(), dims=("obs_id",))
        _ = pm.Data("X_plot", X_plot.flatten(), dims=("plot_id",))
        _ = pm.Data("X_query", X_query.flatten(), dims=("query_id",))

        # ---- Mean developmental trajectory + HSGP deviation ----
        # Logit-linear trend + HSGP, built by the shared helper (gp_utils) so the
        # graph is byte-identical to the previously inlined form: it stores the
        # named `g` and `f_all` Deterministics and the slope/intercept/ell.
        f_all = trend_and_gp(
            cfg_low=context.model_config.p_slope_low_dist,
            cfg_hi=context.model_config.p_slope_hi_dist,
            cfg_ell=context.model_config.ell_unit_dist,
            cfg_eta=context.model_config.eta_dist,
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
        )

        # Slice for obs / plot / query
        f_obs = pm.Deterministic("f_obs", f_all[i_obs0:i_obs1], dims=("obs_id",))
        f_plot = pm.Deterministic("f_plot", f_all[i_plot0:i_plot1], dims=("plot_id",))
        f_query = pm.Deterministic(
            "f_query", f_all[i_query0:i_query1], dims=("query_id",)
        )

        p_obs = pm.Deterministic("p_obs", pm.math.sigmoid(f_obs), dims=("obs_id",))
        _ = pm.Deterministic("p_plot", pm.math.sigmoid(f_plot), dims=("plot_id",))
        _ = pm.Deterministic("p_query", pm.math.sigmoid(f_query), dims=("query_id",))

        # z values for each slice (these are just the standardized ages)
        z_obs = pm.Deterministic(
            "z_obs", X_all_z_data[i_obs0:i_obs1, 0], dims=("obs_id",)
        )
        z_plot = pm.Deterministic(
            "z_plot", X_all_z_data[i_plot0:i_plot1, 0], dims=("plot_id",)
        )
        z_query = pm.Deterministic(
            "z_query", X_all_z_data[i_query0:i_query1, 0], dims=("query_id",)
        )

        # ---- Dispersion / overdispersion ----

        # dispersion parameter κ (kappa) is a function of age (z), with a minimum value and an
        # exponential increase/decrease with age (kappa_min, a_kappa/b_kappa_mag -> b_kappa,
        # shared helper — see models.gp_utils.build_kappa_of_z).
        kappa_of_z = build_kappa_of_z(
            context.model_config.kappa_min_dist,
            context.model_config.a_kappa_dist,
            context.model_config.b_kappa_mag_dist,
        )

        # compute kappa for the observed data points, and use that in the likelihood
        kappa_obs = pm.Deterministic("kappa_obs", kappa_of_z(z_obs), dims="obs_id")

        # compute kappa for the plot and query points (for reporting, not used in likelihood)
        _ = pm.Deterministic("kappa_plot", kappa_of_z(z_plot), dims="plot_id")
        _ = pm.Deterministic("kappa_query", kappa_of_z(z_query), dims="query_id")

        # clip p_obs numerical issues when alpha or beta become extremely close to 0
        p_obs_clip = pm.math.clip(
            p_obs, math_constants.EPSILON, 1 - math_constants.EPSILON
        )

        # compute alpha and beta parameters for the Beta-Binomial likelihood based on p_obs and kappa_obs
        alpha_obs = p_obs_clip * kappa_obs
        beta_obs = (1 - p_obs_clip) * kappa_obs

        # Beta-binomial likelihood
        _ = pm.BetaBinomial(
            "y_obs",
            n=context.model_data.n_trials,
            alpha=alpha_obs,
            beta=beta_obs,
            observed=context.model_data.y_obs,
            dims=("obs_id",),
        )

    variables = pymc_utils.get_variables_dict(model)

    pymc_utils.report_model_summary(model)

    render_model_graph(model, context.reporting.output_dir)

    context.set_model(model, variables)


def prior_predictive_checks(
    context: ModelFitContext,
    outcome_col: str,
    outcome_label: str,
):
    """
    Run prior predictive checks: sample from the prior, and visualise the implied prior predictive distribution.
    """
    with context.model:
        prior_samples = pm.sample_prior_predictive(
            draws=1000,
            random_seed=context.sampling.random_seed,
            compile_kwargs=dict(mode="FAST_COMPILE"),
        )

    # Sample prior functions

    p_plot_samples = (
        prior_samples.prior["p_plot"]
        .stack(sample=("chain", "draw"))
        .transpose("plot_id", "sample")  # (n_plot, n_samples)
    )

    context.set_prior_samples(prior_samples)

    plotting.plot_prior_samples(
        prior_samples.constant_data["X_plot"].values,
        p_plot_samples.values,
        context.analysis_df["age"],
        context.analysis_df[outcome_col],
        n_trials=context.model_data.n_trials,
        n_curves=1000,
        x_label="Age (months)",
        y_label=outcome_label,
        filename="prior_samples",
        output_dir=context.reporting.output_dir,
    )

    y_pred = prior_samples.prior_predictive["y_obs"].stack(sample=("chain", "draw"))
    obs_ages = prior_samples.constant_data["X_obs"].values

    plotting.plot_prior_predictions(
        obs_ages,
        y_pred,
        context.analysis_df["age"],
        context.analysis_df[outcome_col],
        n_trials=context.model_data.n_trials,
        x_label="Age (months)",
        y_label=f"{outcome_label} (predicted)",
        filename="prior_predictions",
        output_dir=context.reporting.output_dir,
    )

    # Prior predictive distribution

    plot_predictive.plot_prior_predictive_checks(
        prior_samples,
        random_seed=context.sampling.random_seed,
        output_dir=context.reporting.output_dir,
        filename="prior_predictive_checks",
    )


def sample(context: ModelFitContext):
    """
    Draw samples from the posterior using MCMC.
    """
    config_table("Sampling configuration", context.sampling)

    with context.model:
        trace = pm.sample(
            context.sampling.draws,
            tune=context.sampling.tune,
            chains=context.sampling.chains,
            cores=context.sampling.cores,
            target_accept=context.sampling.target_accept,
            nuts_sampler="nutpie",
            # rich progress bar segfaults under nutpie's worker threads when
            # stdout is not a TTY (redirected/backgrounded); keep it for
            # interactive terminals only.
            progressbar=sys.stdout.isatty(),
            return_inferencedata=True,
            random_seed=context.sampling.random_seed,
        )

    context.set_trace(trace)


def diagnostics_var_names(model) -> tuple[list[str], list[str]]:
    """Return ``(summary_var_names, gate_var_names)`` for :func:`diagnostics`.

    The summary table and plots show only the scalar (size <= 2) unobserved
    RVs so they stay readable.  The convergence gate must screen every sampled
    parameter, so it additionally covers the vector-valued free RVs — HSGP
    basis coefficients and, in the RE engines, the study/subject random
    intercepts — element-wise.  Vector-valued deterministics (e.g. the GP
    evaluated over the data grid) stay out of both sets: they are derived from
    the free RVs, not sampled.
    """
    summary_var_names = [
        var.name for var in model.unobserved_RVs if var.size.eval() <= 2
    ]
    gate_var_names = summary_var_names + [
        rv.name for rv in model.free_RVs if rv.name not in summary_var_names
    ]
    return summary_var_names, gate_var_names


def diagnostics(
    context: ModelFitContext,
    *,
    extra_trace_var_names: tuple[str, ...] = (),
    loo_var_names: tuple[tuple[str, str], ...] | None = None,
    var_names_fn=None,
    round_to: int = 3,
):
    """
    Run diagnostics on the posterior samples, including convergence diagnostics and posterior predictive checks.

    Shared across every engine (single-outcome, bivariate, trivariate, joint).
    The engines differ only in:

    - ``extra_trace_var_names``: extra (per-outcome kappa) variable names
      appended to the trace plot only, not the summary/pair/posterior-density
      plots.
    - ``loo_var_names``: ``((var_name, heading_label), ...)`` for per-outcome
      LOO-CV (bivariate/trivariate/joint likelihoods are named, e.g.
      ``y_u_obs``). ``None`` runs a single unnamed ``az.loo`` call (the
      single-outcome / study-RE engines, which have exactly one likelihood).
    - ``var_names_fn``: optional ``list[str] -> list[str]`` reordering applied
      before the pair/trace/posterior-density plots only (the joint engine
      prioritises ``psi``/``conc`` first); the summary table always uses the
      unordered ``var_names``.
    - ``round_to``: decimal places for the summary table (joint uses 4 to keep
      ``psi``/``conc`` precision; every other engine uses the default 3).
    """
    # Summary diagnostic statistics

    var_names, gate_var_names = diagnostics_var_names(context.model)
    diagnostics_df = az.summary(
        context.trace,
        var_names=var_names,
        round_to=round_to,
        ci_prob=context.reporting.ci_prob,
        ci_kind="hdi",
    )

    diagnostics_df.to_csv(
        os.path.join(context.reporting.output_dir, "diagnostics.csv"), index=True
    )

    gate_summary = shared_diagnostics.write_diagnostics_summary(
        context.trace, context.reporting.output_dir, var_names=gate_var_names
    )

    dataframe_table(diagnostics_df, title="Posterior diagnostics")
    _report_diagnostic_warnings(gate_summary)

    # Kernel density estimates (KDE) of the joint posterior, and marginals

    plot_var_names = var_names if var_names_fn is None else var_names_fn(var_names)

    pair_plot_var_names = capped_plot_var_names(
        context.trace,
        plot_var_names,
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
        # Drop the SVG: plot_kde_pair always writes both PNG and SVG, but the
        # dense KDE grid produces a multi-megabyte SVG that nothing embeds (the
        # report uses pair_plot.png).
        pair_plot_svg = os.path.join(context.reporting.output_dir, "pair_plot.svg")
        if os.path.exists(pair_plot_svg):
            os.remove(pair_plot_svg)

    # Trace plot

    var_names_ext = plot_var_names + list(extra_trace_var_names)
    trace_var_names = capped_plot_var_names(context.trace, var_names_ext)

    az.plot_trace(
        context.trace,
        var_names=trace_var_names,
        figure_kwargs={"figsize": plot_styles.FIGSIZE_XL},
    )
    plt.savefig(os.path.join(context.reporting.output_dir, "trace_plot.png"), dpi=300)
    context.plots["trace_plot"] = plt.gcf()
    plt.close()

    # Energy transition distribution
    az.plot_energy(
        context.trace,
        figure_kwargs={"figsize": plot_styles.FIGSIZE_XL},
    )
    plt.savefig(os.path.join(context.reporting.output_dir, "energy_plot.png"), dpi=300)
    context.plots["energy_plot"] = plt.gcf()
    plt.close()

    # Posterior densities
    posterior_var_names = capped_plot_var_names(context.trace, plot_var_names)
    az.plot_dist(
        context.trace,
        var_names=posterior_var_names,
        point_estimate="median",
        ci_kind="hdi",
        ci_prob=context.reporting.ci_prob,
    )
    plt.savefig(
        os.path.join(context.reporting.output_dir, "posterior_plot.png"), dpi=300
    )
    context.plots["posterior_plot"] = plt.gcf()
    plt.close()

    # Pareto-smoothed importance sampling

    with context.model:
        trace = pm.compute_log_likelihood(context.trace)

    context.set_trace(trace)

    if loo_var_names is None:
        loocv = az.loo(context.trace)
        context.set_loocv(loocv)
        heading("LOO-CV", style="bold cyan")
        console.print(loocv)
    else:
        loocv_by_name = {
            var_name: az.loo(context.trace, var_name=var_name)
            for var_name, _label in loo_var_names
        }
        context.set_loocv(loocv_by_name)
        for var_name, label in loo_var_names:
            heading(f"LOO-CV — {label}", style="bold cyan")
            console.print(loocv_by_name[var_name])


def sample_posterior_predictive(context: ModelFitContext):
    """
    Sample from the posterior predictive distribution.
    """
    p_plot = context.model_variables["p_plot"]
    p_query = context.model_variables["p_query"]
    kappa_plot = context.model_variables["kappa_plot"]
    kappa_query = context.model_variables["kappa_query"]

    with context.model:
        p_plot_clip = pm.math.clip(
            p_plot, math_constants.EPSILON, 1 - math_constants.EPSILON
        )
        alpha_plot = p_plot_clip * kappa_plot
        beta_plot = (1 - p_plot_clip) * kappa_plot
        pm.BetaBinomial(
            "y_plot",
            n=context.model_data.n_trials,
            alpha=alpha_plot,
            beta=beta_plot,
            dims=("plot_id",),
        )
        p_query_clip = pm.math.clip(
            p_query, math_constants.EPSILON, 1 - math_constants.EPSILON
        )
        alpha_query = p_query_clip * kappa_query
        beta_query = (1 - p_query_clip) * kappa_query
        pm.BetaBinomial(
            "y_query",
            n=context.model_data.n_trials,
            alpha=alpha_query,
            beta=beta_query,
            dims=("query_id",),
        )
        trace = pm.sample_posterior_predictive(
            context.trace,
            var_names=["y_plot", "y_query", "y_obs"],
            extend_inferencedata=True,
            progressbar=sys.stdout.isatty(),
            random_seed=context.sampling.random_seed,
        )

    context.set_trace(trace)

    trace.to_netcdf(os.path.join(context.reporting.output_dir, "trace.nc"))

    sample_data = extract_model_samples(context.trace)
    context.set_model_samples(sample_data)


def posterior_summary(context: ModelFitContext):
    """
    Compute and store the posterior summary table at query ages.
    """

    posterior_summary_df = posterior_analysis.posterior_summary_table(
        context.model_samples.X_query,
        context.model_samples.p_query,
        context.model_samples.y_query,
        n_trials=context.model_data.n_trials,
        hdi_prob=context.reporting.ci_prob,
    )

    dataframe_table(
        posterior_summary_df,
        title="Posterior summary at query ages",
        show_index=False,
    )

    context.dataframes["posterior_summary"] = posterior_summary_df

    posterior_summary_df.to_csv(
        os.path.join(context.reporting.output_dir, "posterior_summary.csv"), index=False
    )


def run_standard_plots(context: ModelFitContext, *, outcome_label: str = "Word count"):
    """
    Run the standard set of posterior predictive plots for single-outcome models.
    """
    plotting.plot_posterior_predictive_count_distributions_by_query_age(
        X_query=context.model_samples.X_query,
        y_query=context.model_samples.y_query,
        n_trials=context.model_data.n_trials,
        hdi_prob=context.reporting.ci_prob,
        output_dir=context.reporting.output_dir,
        filename="posterior_predictive_count_distributions",
        x_label=outcome_label,
    )

    plotting.plot_posterior_predictive_pmf(
        context.model_samples.X_query,
        context.model_samples.X_plot,
        context.model_samples.y_plot,
        context.model_data.n_trials,
        output_dir=context.reporting.output_dir,
        filename="posterior_predictive_pmf",
    )

    plotting.plot_posterior_predictive_cdf(
        context.model_samples.X_query,
        context.model_samples.X_plot,
        context.model_samples.y_plot,
        context.model_data.n_trials,
        output_dir=context.reporting.output_dir,
        filename="posterior_predictive_cdf",
    )

    plotting.plot_posterior_predictive_median_trend(
        context.model_samples.X_plot,
        context.model_samples.y_plot,
        context.model_samples.X_obs,
        context.model_samples.y_obs,
        output_dir=context.reporting.output_dir,
        filename="plot_posterior_predictive_median_trend",
    )

    plotting.plot_posterior_predictive_median_trend(
        context.model_samples.X_plot,
        context.model_samples.y_plot,
        context.model_samples.X_obs,
        context.model_samples.y_obs,
        smooth=True,
        savgol_window_length=15,
        savgol_polyorder=3,
        smooth_intervals=True,
        output_dir=context.reporting.output_dir,
        filename="plot_posterior_predictive_median_trend_smoothed",
    )

    plotting.plot_expected_learning_rate(
        context.model_samples.X_plot,
        context.model_samples.f_plot,
        n_trials=context.model_data.n_trials,
        hdi_prob=context.reporting.ci_prob,
        output_dir=context.reporting.output_dir,
        filename="expected_learning_rate",
    )

    plotting.plot_expected_learning_rate(
        context.model_samples.X_plot,
        context.model_samples.f_plot,
        n_trials=context.model_data.n_trials,
        hdi_prob=context.reporting.ci_prob,
        smooth=True,
        savgol_window_length=15,
        savgol_polyorder=3,
        smooth_intervals=True,
        output_dir=context.reporting.output_dir,
        filename="expected_learning_rate_smoothed",
    )

    plotting.plot_posterior_kappa(
        context.model_samples.X_plot,
        context.model_samples.kappa_plot,
        context.model_samples.X_query,
        context.model_samples.kappa_query,
        n_trials=context.model_data.n_trials,
        hdi_prob=context.reporting.ci_prob,
        output_dir=context.reporting.output_dir,
        filename="posterior_kappa",
    )


def report(context: ModelFitContext):
    """
    Copy output artefacts to the report directory.
    """

    REPORT_OUTPUT_DIR = os.path.join(
        local_env.REPORT_FIGS_DIR, context.reporting.model_label
    )

    if os.path.exists(REPORT_OUTPUT_DIR):
        shutil.rmtree(REPORT_OUTPUT_DIR)

    os.makedirs(REPORT_OUTPUT_DIR, exist_ok=True)

    copied = 0
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
        copied += 1

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

    key_value_table(
        "Artefacts",
        [
            ("Output directory", context.reporting.output_dir),
            ("Report figures directory", REPORT_OUTPUT_DIR),
            ("Figures / tables copied", copied),
            ("Quarto document", model_output_md_dest),
        ],
    )

    console.print(
        f"\n[bold yellow]Render report:[/bold yellow]  "
        f"[blue]quarto render {model_output_md_dest}[/blue]"
    )
    console.print(
        f"[bold yellow]Live preview:[/bold yellow]   "
        f"[blue]quarto preview {model_output_md_dest}[/blue]"
    )


def _plot_and_print_dist(context, dist, name):
    """Plot a prior distribution and print its summary."""
    context.plots[name] = plot_dist.plot_distribution(
        dist, context.reporting.output_dir, name
    )
    summary = dist.summary(mass=context.reporting.ci_prob)
    console.print(f"  [yellow]{name}[/yellow]: {summary}")


_RHAT_WARN = 1.01
_ESS_WARN = 400


def _report_diagnostic_warnings(gate_summary: dict) -> None:
    """Flag MCMC convergence issues recorded in the gate payload.

    ``gate_summary`` is the dict returned by
    ``shared_diagnostics.write_diagnostics_summary``, so the banner covers the
    same (unrounded) R-hat / ESS scan as the gate itself — every free
    parameter, element-wise — rather than the rounded scalar-only summary
    table.
    """
    thresholds = gate_summary.get("thresholds") or {}
    rhat_max = thresholds.get("rhat_max", _RHAT_WARN)
    ess_threshold = thresholds.get("ess_threshold", _ESS_WARN)
    max_rhat = gate_summary.get("max_rhat")
    min_ess = gate_summary.get("min_ess")

    problems: list[str] = []
    rhat_failing = gate_summary.get("rhat_failing") or []
    if rhat_failing:
        detail = f" (max {max_rhat:.3f})" if max_rhat is not None else ""
        problems.append(
            f"{len(rhat_failing)} parameter(s) with r_hat > {rhat_max}{detail}"
        )
    ess_failing = gate_summary.get("ess_failing") or []
    if ess_failing:
        detail = f" (min {min_ess:.0f})" if min_ess is not None else ""
        problems.append(
            f"{len(ess_failing)} parameter(s) with bulk or tail ESS "
            f"< {ess_threshold}{detail}"
        )

    if problems:
        console.print()
        for line in problems:
            console.print(f"[bold yellow]⚠ {line}[/bold yellow]")
    elif max_rhat is not None and min_ess is not None:
        console.print(
            f"[green]✓ r_hat ≤ {rhat_max} and ESS ≥ {ess_threshold} across all "
            "free parameters (including vector-valued).[/green]"
        )
    # When the gate's R-hat/ESS scan itself failed (both values None),
    # write_diagnostics_summary has already reported it; claim nothing here.


def prepare_univariate_data(
    context: ModelFitContext,
    definition: UnivariateModelDefinition,
):
    """Load and prepare data for a univariate model from its definition."""
    y_col = definition.outcome.value
    df = vocab_data_utils.load_data(
        population=definition.population,
        columns=["age", y_col],
        sample_fraction=definition.sample_fraction,
        random_seed=definition.random_seed,
    )
    analysis_df = df[["age", y_col]].dropna()

    desc = descriptive_stats.describe_all(analysis_df, alpha=0.05)

    key_value_table(
        "Data",
        [
            ("Population", definition.population.name),
            ("Outcome column", y_col),
            ("Rows after NA drop", len(analysis_df)),
            ("Sample fraction", definition.sample_fraction),
        ],
    )
    dataframe_table(desc, title="Descriptive statistics")

    X_obs = np.asarray(analysis_df["age"], dtype=float).reshape(-1, 1)
    y_obs = np.asarray(analysis_df[y_col], dtype=int)

    data = model_data.BinomialModelData(
        X_obs=X_obs, y_obs=y_obs, n_trials=definition.n_trials
    )

    context.set_model_data(data, analysis_df)
    context.dataframes["descriptive_stats"] = desc

    desc.to_csv(
        os.path.join(context.reporting.output_dir, "descriptive_statistics.csv"),
        index=True,
    )


def configure_univariate_priors(
    context: ModelFitContext,
    definition: UnivariateModelDefinition,
):
    """Configure priors and hyperparameters from a univariate model definition."""
    ell_unit_dist = pz.Beta(alpha=definition.ell_unit_alpha, beta=definition.ell_unit_beta)
    _plot_and_print_dist(context, ell_unit_dist, "ell_unit_dist")

    eta_dist = pz.HalfNormal(sigma=definition.eta_sigma)
    _plot_and_print_dist(context, eta_dist, "eta_dist")

    p_slope_low_dist = pz.Beta(alpha=definition.p_slope_low_alpha, beta=definition.p_slope_low_beta)
    _plot_and_print_dist(context, p_slope_low_dist, "p_slope_low_dist")

    p_slope_hi_dist = pz.Beta(alpha=definition.p_slope_hi_alpha, beta=definition.p_slope_hi_beta)
    _plot_and_print_dist(context, p_slope_hi_dist, "p_slope_hi_dist")

    kp = definition.kappa
    kappa_min_dist = pz.LogNormal(mu=kp.kappa_min_mu, sigma=kp.kappa_min_sigma)
    _plot_and_print_dist(context, kappa_min_dist, "kappa_min_dist")

    a_kappa_dist = pz.Normal(mu=kp.a_kappa_mu, sigma=kp.a_kappa_sigma)
    _plot_and_print_dist(context, a_kappa_dist, "a_kappa_dist")

    b_kappa_mag_dist = pz.HalfNormal(sigma=kp.b_kappa_mag_sigma)
    _plot_and_print_dist(context, b_kappa_mag_dist, "b_kappa_mag_dist")

    config = ModelConfiguration(
        slope_anchors=definition.slope_anchors,
        ell_months_range=definition.ell_months_range,
        p_slope_low_dist=p_slope_low_dist,
        p_slope_hi_dist=p_slope_hi_dist,
        ell_unit_dist=ell_unit_dist,
        eta_dist=eta_dist,
        kappa_min_dist=kappa_min_dist,
        a_kappa_dist=a_kappa_dist,
        b_kappa_mag_dist=b_kappa_mag_dist,
        n_plot=definition.n_plot,
        ages_query=definition.ages_query,
    )

    context.set_model_config(config)


def run_fit_pipeline(
    config: str,
    definition,
    *,
    stages: list[tuple[str, Callable[[ModelFitContext], None]]],
) -> ModelFitContext:
    """Shared fit-pipeline scaffold used by every engine's ``fit_*_model``.

    Every engine's orchestration function differed only in *which* stage
    functions it ran (data prep / prior config / model build / predictive
    sampling / plots are all engine-specific); the banner, environment and
    package reporting, ``ModelFitContext`` construction, output-directory
    clearing, timed-section bookkeeping and final summary were six identical
    copies. ``stages`` is the ordered ``(section_name, fn)`` list, where each
    ``fn`` takes the freshly-created ``context`` (typically a closure binding
    the engine's ``definition``).
    """
    run_banner(definition.banner, subtitle=f"sampling config: {config}")

    env_info.report_environment_info()

    console.print()
    package_metadata.report_package_versions(PACKAGE_LIST)

    context = ModelFitContext(
        reporting=reporting.ReportingConfiguration(
            model_name=definition.model_id,
            config_name=definition.config_name,
            output_root_dir=local_env.output_root(),
            ci_prob=0.90,
            interval_kind="hdi",
        ),
        sampling=sampling.get_sampling_configuration(config),
    )

    if os.path.exists(context.reporting.output_dir):
        shutil.rmtree(context.reporting.output_dir)

    os.makedirs(context.reporting.output_dir, exist_ok=True)

    timings = context.timings
    run_started = time.perf_counter()

    for name, fn in stages:
        with section(name, timings=timings):
            fn(context)

    pipeline_summary(
        f"Pipeline summary — {context.reporting.model_label}", timings
    )
    console.print(
        f"[dim]Total wall time: "
        f"{vg_reporting.format_duration(time.perf_counter() - run_started)}[/dim]"
    )

    return context


def fit_single_outcome_model(
    config: str,
    definition: UnivariateModelDefinition,
) -> ModelFitContext:
    """
    Shared fit pipeline for single-outcome models (VG01-VG04).
    """
    y_col = definition.outcome.value
    outcome_label = definition.outcome_label

    return run_fit_pipeline(
        config,
        definition,
        stages=[
            ("Prepare data", lambda ctx: prepare_univariate_data(ctx, definition)),
            (
                "Priors and hyperparameters",
                lambda ctx: configure_univariate_priors(ctx, definition),
            ),
            ("Model definition and initialisation", build_model),
            (
                "Prior predictive checks",
                lambda ctx: prior_predictive_checks(
                    ctx, outcome_col=y_col, outcome_label=outcome_label
                ),
            ),
            ("Posterior sampling", sample),
            ("Diagnostics", diagnostics),
            ("Posterior predictions", sample_posterior_predictive),
            ("Posterior summary", posterior_summary),
            (
                "Plots",
                lambda ctx: run_standard_plots(ctx, outcome_label=outcome_label),
            ),
            ("Report", report),
        ],
    )
