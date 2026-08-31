# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Shared dataclasses and pipeline functions for the vocabulary growth model family.
"""

import json
import os
import platform
import shutil
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from importlib import metadata as importlib_metadata
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
import vocab_growth.reporting_ages as reporting_ages
from vocab_growth.analysis_frames import analysis_frame_hash
from vocab_growth.fit_artifacts import (
    ACCEPTED_EXCEPTION_KEY,
    CONVERGENCE_CAVEATS_FILENAME,
    CONVERGENCE_FAILURE_FILENAME,
    DIAGNOSTICS_SUMMARY_FILENAME,
    FIT_MANIFEST_FILENAME,
    NOT_SAMPLED_ATTR,
    SAMPLED_PARAMETERS_ATTR,
    TracePersistence,
    accepted_rhat_exception,
    convergence_caveats,
    create_staging_root,
    git_metadata,
    is_reporting_quality_config,
    normalise_for_json,
    promote_staged_fit,
    retain_failed_fit,
    sampled_variable_names,
    save_trace,
    source_data_hash,
    unsampled_deterministic_names,
    write_fit_state,
    write_json_atomic,
)
from vocab_growth.loo_reff import sampled_parameter_reff
from vocab_growth.models import fit_identity
from vocab_growth.models.build_utils import (
    construct_age_grids,
    require_integral_counts,
    slope_anchor_logit_coeffs,
    standardize_ages,
    standardize_anchor_ages,
    validate_ell_bounds,
)
from vocab_growth.models.calibration import write_trace_calibration
from vocab_growth.models.definitions import (
    KappaAnchorPriorParams,
    UnivariateModelDefinition,
)
from vocab_growth.models.diagnostics_utils import capped_plot_var_names
from vocab_growth.models.gp_utils import (
    GPGrid,
    build_kappa_of_z,
    build_kappa_of_z_anchored,
    trend_and_gp,
)
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
    ages_query: tuple[int, ...]
    """Ages in months for querying the model."""

    def __post_init__(self) -> None:
        if len(self.slope_anchors) != 2:
            raise ValueError("slope_anchors must be a tuple of two float values.")
        if len(self.ell_months_range) != 2:
            raise ValueError("ell_months_range must be a tuple of two int values.")
        if self.n_plot <= 0:
            raise ValueError("n_plot must be a positive integer.")
        # A tuple since issue #273 froze the definitions, but any non-empty
        # ordered collection of ages is usable here; the check is on emptiness,
        # not on the container, so a caller passing a list is not refused for a
        # reason that has nothing to do with the model.
        if isinstance(self.ages_query, (str, bytes)) or len(self.ages_query) == 0:
            raise ValueError("ages_query must be a non-empty sequence of integers.")


@dataclass(frozen=True)
class AnchoredKappaPriors:
    """Resolved two-anchor dispersion priors: the distributions and their ages.

    The configuration-side counterpart of
    :class:`~vocab_growth.models.definitions.KappaAnchorPriorParams`. Held as one
    object rather than as loose fields so a model configuration carries either
    this or the legacy ``a_kappa`` / ``b_kappa_mag`` pair, never a half-specified
    mixture of the two.
    """

    kappa_min_dist: Continuous
    """Prior for the dispersion asymptote (kappa_min).

    The end of the age range it applies at follows the sign of the derived
    ``b_kappa``: falling dispersion makes it the old-age floor, rising dispersion
    the young-age one (VG13's is 30, not 3, for that reason).
    """
    excess_young_dist: Continuous
    """Prior for the age term above the asymptote at the young anchor age."""
    excess_old_dist: Continuous
    """Prior for the age term above the asymptote at the old anchor age."""
    anchor_ages: tuple[float, float]
    """Reference ages (months), ordered (young, old)."""


@dataclass
class ModelConfiguration(BaseModelConfiguration):
    """Configuration for a single-outcome model (VG01-VG04).

    Dispersion is specified in exactly one of two ways: the legacy
    ``kappa_min_dist`` / ``a_kappa_dist`` / ``b_kappa_mag_dist`` triple, or
    ``kappa_anchored``. ``__post_init__`` rejects anything else.
    """

    p_slope_low_dist: Continuous
    """Prior distribution for the mean proportion of the outcome at the lower slope anchor."""
    p_slope_hi_dist: Continuous
    """Prior distribution for the mean proportion of the outcome at the upper slope anchor."""
    ell_unit_dist: Continuous
    """Prior distribution for the unit length-scale parameter (ell_unit)."""
    eta_dist: Continuous
    """Prior distribution for the GP amplitude (eta)."""
    kappa_min_dist: Continuous | None = None
    """Prior distribution for the minimum kappa value (kappa_min); legacy form only."""
    a_kappa_dist: Continuous | None = None
    """Prior distribution for the age slope of kappa (a_kappa); legacy form only."""
    b_kappa_mag_dist: Continuous | None = None
    """Prior distribution for the magnitude of the kappa parameter; legacy form only."""
    kappa_anchored: AnchoredKappaPriors | None = None
    """Two-anchor dispersion priors, in place of the three fields above."""
    report_max_age_understood: int | None = None
    """Highest query age reported, for a comprehension model. Reporting only."""
    variance_partition_total_dist: Continuous | None = None
    """Prior for the total logit-scale scatter at the young kappa anchor.

    Set together with ``variance_partition_share_dist`` when the model splits one
    scatter budget between the subject-effect scale and dispersion rather than
    giving them competing priors. See
    :func:`~vocab_growth.models.gp_utils.build_variance_partition`."""
    variance_partition_share_dist: Continuous | None = None
    """Prior for the subject share of that scatter."""
    variance_partition_reference_proportion: float | None = None
    """Fixed proportion at which concentration is converted to logit-scale variance."""

    def __post_init__(self) -> None:
        super().__post_init__()
        validate_kappa_fields(self)
        partition = (
            self.variance_partition_total_dist,
            self.variance_partition_share_dist,
            self.variance_partition_reference_proportion,
        )
        if any(p is not None for p in partition) and not all(
            p is not None for p in partition
        ):
            raise ValueError(
                "The variance-partition fields must be set together: "
                "variance_partition_total_dist, variance_partition_share_dist "
                "and variance_partition_reference_proportion."
            )
        if partition[0] is not None and self.kappa_anchored is None:
            raise ValueError(
                "The variance partition allocates the two-anchor kappa form's "
                "young anchor, so kappa_anchored must be configured."
            )

    @property
    def has_variance_partition(self) -> bool:
        return self.variance_partition_total_dist is not None


def _legacy_kappa_dists(config, suffix):
    """The legacy triple for one outcome, by the engine's field-naming convention.

    Single-outcome configurations use ``kappa_min_dist``; the joint ones append
    the outcome to the same stems (``kappa_min_u_dist``), so one suffix drives
    both the field lookup here and the RV names in the graph.
    """
    return (
        getattr(config, f"kappa_min{suffix}_dist", None),
        getattr(config, f"a_kappa{suffix}_dist", None),
        getattr(config, f"b_kappa_mag{suffix}_dist", None),
    )


def validate_kappa_fields(config, suffixes=("",)) -> None:
    """Reject a configuration that half-specifies a dispersion form.

    Each outcome carries exactly one of the two parameterisations. Outcomes are
    independent of each other — a joint model may anchor ``kappa_u`` while
    ``kappa_s`` stays on the legacy form — but neither may be partial, and no
    outcome may carry both.
    """
    for suffix in suffixes:
        legacy = _legacy_kappa_dists(config, suffix)
        label = f"kappa{suffix}"
        if getattr(config, f"kappa_anchored{suffix}", None) is not None:
            if any(dist is not None for dist in legacy):
                raise ValueError(
                    f"kappa_anchored{suffix} cannot be combined with the legacy "
                    f"{label} kappa_min / a_kappa / b_kappa_mag distributions."
                )
        elif any(dist is None for dist in legacy):
            raise ValueError(
                f"the legacy {label} form needs all of kappa_min{suffix}_dist, "
                f"a_kappa{suffix}_dist and b_kappa_mag{suffix}_dist; pass "
                f"kappa_anchored{suffix} instead."
            )


def kappa_prior_rows(config, suffix="") -> list[tuple[str, object]]:
    """Rows for the "Priors" table, naming whichever kappa form the model uses.

    The anchored rows carry their reference ages, because the age is part of what
    the prior means — ``kappa_excess_young`` on its own says nothing.
    """
    anchored = getattr(config, f"kappa_anchored{suffix}", None)
    if anchored is None:
        kappa_min_dist, a_kappa_dist, b_kappa_mag_dist = _legacy_kappa_dists(
            config, suffix
        )
        return [
            (f"kappa_min{suffix}", kappa_min_dist),
            (f"a_kappa{suffix}", a_kappa_dist),
            (f"b_kappa_mag{suffix}", b_kappa_mag_dist),
        ]
    young_age, old_age = anchored.anchor_ages
    return [
        (f"kappa_min{suffix}", anchored.kappa_min_dist),
        (
            f"kappa_excess_young{suffix} ({young_age:g} mo)",
            anchored.excess_young_dist,
        ),
        (f"kappa_excess_old{suffix} ({old_age:g} mo)", anchored.excess_old_dist),
    ]


def kappa_anchor_derived_rows(config, *, X_obs_mean, X_obs_std, suffix=""):
    """Rows for the "Derived quantities" table, or none if unanchored.

    The z positions are what the graph actually uses, and they move with the
    pool's age distribution even though the ages do not, so they are worth
    printing next to the slope anchors.
    """
    anchored = getattr(config, f"kappa_anchored{suffix}", None)
    if anchored is None:
        return []
    return [
        (f"Kappa anchors{suffix} (months)", anchored.anchor_ages),
        (
            f"Kappa anchors{suffix} (z-score)",
            standardize_anchor_ages(
                anchored.anchor_ages, X_obs_mean=X_obs_mean, X_obs_std=X_obs_std
            ),
        ),
    ]


def build_kappa_for_config(
    config,
    *,
    X_obs_mean,
    X_obs_std,
    suffix="",
    excess_young_value=None,
    hold_constant=False,
):
    """Create the dispersion RVs for whichever kappa form ``config`` carries.

    The single point where the two parameterisations diverge inside a model
    graph. The anchored form needs the observed-age standardisation to place its
    reference ages on the z scale; the legacy form does not, and is emitted
    unchanged. ``suffix`` selects the outcome for the joint engines ("_u", "_s",
    "_sign") and names the resulting variables.

    ``excess_young_value`` supplies the young anchor from outside, for the
    variance-partition reparameterisation. It is only meaningful for the anchored
    form, so pairing it with the legacy one is rejected rather than silently
    ignored. ``hold_constant`` (Proposal A1) is rejected on the same grounds: the
    legacy form's slope is a free ``b_kappa_mag``, so pinning it flat there would
    be a different change to a different parameterisation, and silently ignoring
    the flag would let an A1 variant report a dispersion trajectory it believes
    it has switched off.
    """
    anchored = getattr(config, f"kappa_anchored{suffix}", None)
    if anchored is None:
        if excess_young_value is not None:
            raise ValueError(
                "excess_young_value requires the two-anchor kappa form; the "
                f"legacy parameterisation is configured for suffix {suffix!r}."
            )
        if hold_constant:
            raise ValueError(
                "hold_constant requires the two-anchor kappa form; the legacy "
                f"parameterisation is configured for suffix {suffix!r}."
            )
        kappa_min_dist, a_kappa_dist, b_kappa_mag_dist = _legacy_kappa_dists(
            config, suffix
        )
        return build_kappa_of_z(
            kappa_min_dist, a_kappa_dist, b_kappa_mag_dist, suffix=suffix
        )
    return build_kappa_of_z_anchored(
        anchored.kappa_min_dist,
        anchored.excess_young_dist,
        anchored.excess_old_dist,
        anchor_z=standardize_anchor_ages(
            anchored.anchor_ages, X_obs_mean=X_obs_mean, X_obs_std=X_obs_std
        ),
        suffix=suffix,
        excess_young_value=excess_young_value,
        hold_constant=hold_constant,
    )


@dataclass
class ModelSamples:
    """Posterior, predictive and constant quantities the reporting stages read.

    Carries the plot- and query-grid quantities only. The observation-level
    posterior (``f_obs``, ``p_obs``, ``z_obs``) used to be extracted here as
    well, at ``n_obs x n_samples`` each, and nothing read it; since 2026-08-23
    those variables are not stored by the sampler at all (see
    :func:`vocab_growth.fit_artifacts.sampled_variable_names`).
    """

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
    f_plot: np.ndarray
    """Posterior samples of the latent linear predictor f for the plot points: (n_plot, n_samples)"""
    f_query: np.ndarray
    """Posterior samples of the latent linear predictor f for the query points: (n_query, n_samples)"""
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
    sampling_config_name: str = "unknown"
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
    X_gp_domain_z,
    ell_range_z,
):
    """
    Compute the HSGP boundary ``L`` and basis size ``M`` for the age kernel.

    Parameters
    ----------
    X_gp_domain_z : np.ndarray
        Standardised lower and upper endpoints of the declared HSGP age domain,
        shape (2, 1). Reporting query ages are deliberately excluded.
    ell_range_z : tuple of float
        Length-scale range in z-score scale.

    Returns
    -------
    tuple[list[float], list[int]]
        ``(L, M)`` where ``L`` is the HSGP boundary and ``M`` is the basis
        size, each wrapped in a single-element list (one per input dim).
    """
    x_min = float(np.min(X_gp_domain_z))
    x_max = float(np.max(X_gp_domain_z))

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
        X_plot_z=X_plot_z,
        X_query_z=X_query_z,
        f_plot=f_plot,
        f_query=f_query,
        p_plot=p_plot,
        p_query=p_query,
        y_obs=y_obs,
        y_plot=y_plot,
        y_query=y_query,
        kappa_plot=kappa_plot,
        kappa_query=kappa_query,
    )

    return model_samples


def build_model(
    context: ModelFitContext,
    definition: UnivariateModelDefinition,
):
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
            *kappa_prior_rows(context.model_config),
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
        gp_domain_months=definition.gp_domain_months,
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
        grids.X_gp_domain_z,
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
            *kappa_anchor_derived_rows(
                context.model_config, X_obs_mean=X_obs_mean, X_obs_std=X_obs_std
            ),
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
                # Pin the HSGP basis centre to the declared domain's midpoint so
                # a reporting-query change cannot move the approximation's
                # accuracy region (#234). For every current model of record the
                # pinned value equals the lazily computed one, so this changes
                # no fitted graph.
                x_center_z=float(np.mean(grids.X_gp_domain_z)),
            ),
            store_deterministic=True,
            latent_name="f_all",
            # getattr: the field lives on the joint definition classes only. The
            # 2026-08-04 mean-extrapolation fix was applied to the joint models
            # and never reached the univariate ones, which are exactly the models
            # where `eta` still presses its prior (VG01, VG03, VG11, VG12). See
            # notes/202608042030-q-mean-extrapolation.md.
            clamp_above_hi=getattr(definition, "clamp_mean_above_hi_anchor", False),
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
        # exponential increase/decrease with age. Either parameterisation of that curve —
        # the legacy a_kappa/b_kappa_mag pair or two age anchors — is dispatched by
        # build_kappa_for_config; both end at models.gp_utils.make_kappa_of_z.
        kappa_of_z = build_kappa_for_config(
            context.model_config, X_obs_mean=X_obs_mean, X_obs_std=X_obs_std
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
        # Deliberately no ``mode="FAST_COMPILE"``. It compiles quickly and then
        # executes slowly, across the whole graph, and this stage is a *fixed*
        # cost that does not scale with the sampling tier -- 16m33s of VG12's
        # 3h26m hightune, and a third of a short fit. Measured on VG01 at
        # 20-40x slower than the default optimising mode for the same draws:
        # free variables agree to 7e-16 and deterministics to 3e-15 at one
        # seed, which is float association under the graph rewrites rather
        # than a different distribution. All four engines match; see
        # notes/202608251100-prior-predictive-compile-mode.md.
        prior_samples = pm.sample_prior_predictive(
            draws=1000,
            random_seed=context.sampling.random_seed,
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


def sample(context: ModelFitContext, *, store_observation_deterministics: bool = False):
    """
    Draw samples from the posterior using MCMC.

    The observation-sized deterministics (``f_obs``, ``p_obs``, ``kappa_obs``,
    their per-outcome counterparts and the concatenated ``*_all`` grids) are not
    stored: ``pm.sample`` is given ``var_names`` from
    :func:`vocab_growth.fit_artifacts.sampled_variable_names`, and nutpie then never
    evaluates the rest. The graph is untouched, so the draws are identical to a
    fit that stored them, and the deterministics remain in the model for
    :mod:`vocab_growth.posterior_recompute` to rebuild on demand. Nothing in the
    fit pipeline reads them; storing them was what made fit memory scale as
    ``n_obs x draws`` (``notes/202608050900-td-hierarchical-geometry.md`` §10).

    ``store_observation_deterministics=True`` restores the old behaviour, for a
    caller that will read them across every draw anyway and would gain nothing
    from a second pass (``scripts/kfold_loso.py``). The trace records which
    deterministics were left out in its posterior attributes either way.
    """
    config_table("Sampling configuration", context.sampling)

    if store_observation_deterministics:
        var_names = None
        not_sampled: list[str] = []
    else:
        var_names = sampled_variable_names(context.model)
        not_sampled = unsampled_deterministic_names(context.model)
        if not_sampled:
            console.print(
                f"Not storing {len(not_sampled)} observation-sized deterministic(s): "
                + ", ".join(not_sampled)
            )

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
            var_names=var_names,
        )

    # Stored as JSON strings rather than lists: netCDF attribute support for
    # string arrays differs between backends, and a single string round-trips
    # through every one of them. The sampled-parameters record is what lets a
    # reader of the stored trace pin PSIS-LOO's relative efficiency to the
    # parameters the sampler moved (loo_reff) without rebuilding the model.
    trace.posterior.attrs[NOT_SAMPLED_ATTR] = json.dumps(not_sampled)
    trace.posterior.attrs[SAMPLED_PARAMETERS_ATTR] = json.dumps(
        [rv.name for rv in context.model.free_RVs]
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


LOO_SUMMARY_FILENAME = "loo_summary.csv"


def emit_loo_summary(
    loo_by_label: dict,
    dropped_by_label: dict[str, int],
    output_dir: str,
    *,
    reff: float | None = None,
) -> pd.DataFrame:
    """Persist the LOO-CV result, which every fit computed and then discarded.

    ``elpd`` was printed to the console and dropped on the floor, while the
    predictive-calibration section of every model report points the reader at
    leave-one-out as the out-of-sample counterpart to its in-sample checks. The
    number the reader was sent to find did not exist anywhere they could reach.

    The Pareto k counts travel with the estimate because without them the
    estimate cannot be judged: PSIS-LOO is only trustworthy where the importance
    weights are well behaved, and a handful of observations above the threshold
    is the signal that a reported ``elpd`` is optimistic. ArviZ's own threshold
    (``good_k``, sample-size dependent) is recorded rather than a hard-coded
    0.7, so the bands mean the same thing across fits of different lengths.

    ``reff`` is the relative efficiency every LOO here was computed with — pinned
    to the sampled parameters (:mod:`vocab_growth.loo_reff`) — recorded because
    the Pareto-k bands depend on it and a reader comparing fits should be able
    to see they share the convention.
    """
    rows = []
    for label, loo in loo_by_label.items():
        pareto_k = getattr(loo, "pareto_k", None)
        good_k = getattr(loo, "good_k", None)
        counts = {"good": None, "bad": None, "very_bad": None}
        if pareto_k is not None and good_k is not None:
            values = np.asarray(pareto_k).ravel()
            values = values[np.isfinite(values)]
            counts = {
                "good": int((values <= float(good_k)).sum()),
                "bad": int(((values > float(good_k)) & (values <= 1.0)).sum()),
                "very_bad": int((values > 1.0).sum()),
            }
        rows.append(
            {
                "outcome": label,
                "elpd_loo": float(getattr(loo, "elpd", np.nan)),
                "se": float(getattr(loo, "se", np.nan)),
                "p_loo": float(getattr(loo, "p", np.nan)),
                "n_data_points": int(getattr(loo, "n_data_points", 0)),
                "n_samples": int(getattr(loo, "n_samples", 0)),
                "n_dropped_degenerate": int(dropped_by_label.get(label, 0)),
                "pareto_k_good": counts["good"],
                "pareto_k_bad": counts["bad"],
                "pareto_k_very_bad": counts["very_bad"],
                "good_k_threshold": None if good_k is None else float(good_k),
                "scale": str(getattr(loo, "scale", "log")),
                "reff": None if reff is None else float(reff),
            }
        )

    frame = pd.DataFrame(rows)
    frame.to_csv(os.path.join(output_dir, LOO_SUMMARY_FILENAME), index=False)
    return frame


def _loo_dropping_degenerate(idata, var_name=None, *, reff=None):
    """Compute PSIS-LOO, excluding observations with a constant pointwise
    log-likelihood.

    The nested joint likelihood models a paired outcome (spoken/signed)
    conditionally on the observed *understood* count as its denominator, so a
    row with ``understood == 0`` gives that outcome a structurally constant
    (``n = 0``) log-likelihood across every draw. PSIS requires the pointwise
    log-likelihood to vary across draws — arviz-stats raises
    ``"All tail values are the same"`` on such degenerate points. Those
    observations carry no information about the paired outcome, so they are
    dropped from that outcome's LOO. Because the drop is deterministic (it keys
    off ``n = 0`` alone), every joint model excludes the *same* observations,
    keeping the per-outcome elpd comparable across models (``loo_compare``).

    Returns ``(loo, n_dropped)``.
    """
    ll = idata.log_likelihood
    name = var_name if var_name is not None else list(ll.data_vars)[0]
    da = ll[name]
    sample_dims = [d for d in ("chain", "draw") if d in da.dims]
    obs_dims = [d for d in da.dims if d not in sample_dims]
    n_dropped = 0
    loo_source = idata
    if len(obs_dims) == 1:
        # A structurally constant (n = 0) log-likelihood is numerically ~0 with
        # only floating-point noise, so its across-draw variance is ~1e-33, not
        # exactly zero. Genuinely informative observations vary by orders of
        # magnitude more (variance >> 1e-6), so 1e-12 cleanly separates the two.
        keep = (da.var(dim=sample_dims) > 1e-12).values
        n_dropped = int(keep.size) - int(keep.sum())
        if n_dropped:
            # ``idata`` is an xarray ``DataTree`` (arviz >= 1.2); ``copy`` is
            # shallow by default, so the (large) posterior/sample_stats groups
            # share buffers with the original rather than being duplicated —
            # only the tree structure is copied. Reassigning the sliced
            # (``isel`` view) log-likelihood therefore does not mutate the
            # caller's ``idata``, so repeated per-outcome calls stay independent.
            loo_source = idata.copy(deep=False)
            loo_source["log_likelihood"] = loo_source["log_likelihood"].isel(
                {obs_dims[0]: keep}
            )
    # Compute LOO for the resolved ``name`` (not the raw ``var_name``): when
    # ``var_name`` is None, ``az.loo`` rejects a log-likelihood group holding
    # more than one array, so keep the LOO target identical to the drop target.
    return az.loo(loo_source, var_name=name, reff=reff), n_dropped


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
        ci_kind=context.reporting.interval_kind,
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
        ci_kind=context.reporting.interval_kind,
        ci_prob=context.reporting.ci_prob,
    )
    plt.savefig(
        os.path.join(context.reporting.output_dir, "posterior_plot.png"), dpi=300
    )
    context.plots["posterior_plot"] = plt.gcf()
    plt.close()

    try:
        enforce_convergence_gate(
            gate_summary,
            sampling_config_name=context.sampling_config_name,
            output_dir=context.reporting.output_dir,
            model_id=context.reporting.model_name,
        )
    except ConvergenceGateError:
        # A failed reporting fit never reaches posterior-predictive sampling,
        # which normally writes trace.nc. Preserve the posterior here so the
        # convergence failure can be investigated and reproduced. Pinned to
        # `full` regardless of the configured tier: this trace exists only to be
        # investigated, and a post-mortem should not have to work around a
        # storage policy.
        save_trace(
            context.trace,
            context.reporting.output_dir,
            persistence=TracePersistence.FULL,
        )
        raise

    # Pareto-smoothed importance sampling

    with context.model:
        trace = pm.compute_log_likelihood(context.trace)

    context.set_trace(trace)

    # Relative efficiency over the sampled parameters, not over whatever the
    # posterior group happens to store (loo_reff explains the difference and
    # the decision behind it). Computed once and shared by every LOO below.
    reff = sampled_parameter_reff(
        context.trace, names=[rv.name for rv in context.model.free_RVs]
    )
    console.print(f"PSIS-LOO relative efficiency (sampled parameters): {reff:.4f}")

    dropped_by_label: dict[str, int] = {}
    if loo_var_names is None:
        loocv, n_dropped = _loo_dropping_degenerate(context.trace, reff=reff)
        dropped_by_label["all"] = n_dropped
        if n_dropped:
            console.print(
                f"[yellow]LOO: dropped {n_dropped} degenerate "
                f"(constant log-likelihood) observation(s).[/yellow]"
            )
        context.set_loocv(loocv)
        heading("LOO-CV", style="bold cyan")
        console.print(loocv)
        emit_loo_summary(
            {"all": loocv}, dropped_by_label, context.reporting.output_dir, reff=reff
        )
    else:
        loocv_by_name = {}
        by_label = {}
        for var_name, label in loo_var_names:
            loocv_by_name[var_name], n_dropped = _loo_dropping_degenerate(
                context.trace, var_name=var_name, reff=reff
            )
            by_label[label] = loocv_by_name[var_name]
            dropped_by_label[label] = n_dropped
            if n_dropped:
                console.print(
                    f"[yellow]LOO ({label}): dropped {n_dropped} degenerate "
                    f"(constant log-likelihood) observation(s).[/yellow]"
                )
        context.set_loocv(loocv_by_name)
        for var_name, label in loo_var_names:
            heading(f"LOO-CV — {label}", style="bold cyan")
            console.print(loocv_by_name[var_name])
        emit_loo_summary(
            by_label, dropped_by_label, context.reporting.output_dir, reff=reff
        )


def sample_posterior_predictive(
    context: ModelFitContext,
    definition: UnivariateModelDefinition | None = None,
):
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

    calibration_df = write_trace_calibration(
        trace,
        context.analysis_df,
        context.reporting.output_dir,
        (
            (
                definition.outcome.value if definition is not None else "outcome",
                "y_obs",
                None,
            ),
        ),
    )
    context.dataframes["posterior_predictive_calibration"] = calibration_df

    save_trace(trace, context.reporting.output_dir)

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
        ci_prob=context.reporting.ci_prob,
        interval_kind=context.reporting.interval_kind,
    )

    # A comprehension model may report a shorter grid than its query ages, where
    # the comprehension evidence stops short of them. Validation confines this to
    # an understood-outcome model, so it cannot be a silent no-op on a spoken one.
    posterior_summary_df = posterior_analysis.trim_reported_ages(
        posterior_summary_df,
        getattr(context.model_config, "report_max_age_understood", None),
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

    emit_monthly_summary(
        output_dir=context.reporting.output_dir,
        X_plot=context.model_samples.X_plot,
        p_plot=context.model_samples.p_plot,
        y_plot=context.model_samples.y_plot,
        X_obs=context.model_samples.X_obs,
        n_trials=context.model_data.n_trials,
        ci_prob=context.reporting.ci_prob,
        interval_kind=context.reporting.interval_kind,
        dataframes=context.dataframes,
        plots=context.plots,
    )


def emit_monthly_summary(
    *,
    output_dir: str,
    X_plot,
    p_plot,
    y_plot,
    X_obs,
    n_trials: int,
    ci_prob: float,
    interval_kind: str = "eti",
    suffix: str | None = None,
    outcome_label: str = "words",
    y_label: str = "Word count",
    dataframes: dict | None = None,
    plots: dict | None = None,
    max_age_months: float | None = None,
):
    """Write the whole-month summary table and its expected-count figure.

    The report continues to quote the 6-monthly ``ages_query`` table; this is the
    finer companion, one row per whole month, for readers who need monthly
    resolution. It is derived from the plot grid, so it adds no query ages to the
    model graph and leaves the ``query_id`` dimension the report and comparisons
    consume untouched (see
    :func:`vocab_growth.posterior_analysis.monthly_summary_table`).

    ``y_plot`` may be None where an engine draws no predictive counts on the plot
    grid; the table and figure then cover the expected count only.
    """
    stem = "posterior_summary_monthly" if suffix is None else f"posterior_summary_monthly_{suffix}"
    monthly = posterior_analysis.monthly_summary_table(
        X_plot,
        p_plot,
        y_plot,
        n_trials=n_trials,
        X_obs=X_obs,
        ci_prob=ci_prob,
        interval_kind=interval_kind,
    )
    # One trim serves both artefacts. The table and the figure are built from
    # this frame, so capping here is why they cannot disagree -- the same reason
    # the frame is shared in the first place. See
    # :mod:`vocab_growth.reporting_ages` for which cap each outcome takes.
    monthly = posterior_analysis.trim_reported_ages(monthly, max_age_months)
    monthly.to_csv(os.path.join(output_dir, f"{stem}.csv"), index=False)
    if dataframes is not None:
        dataframes[stem] = monthly

    figure_stem = (
        "expected_counts_by_month" if suffix is None else f"expected_counts_by_month_{suffix}"
    )
    fig = plotting.plot_expected_counts_by_month(
        monthly,
        n_trials=n_trials,
        ci_prob=ci_prob,
        output_dir=output_dir,
        filename=figure_stem,
        y_label=y_label,
        outcome_label=outcome_label,
    )
    if plots is not None:
        plots[figure_stem] = fig
    plt.close(fig)
    return monthly


def run_standard_plots(
    context: ModelFitContext,
    *,
    outcome_label: str = "Word count",
    quantity: reporting_ages.ReportedQuantity | None = None,
):
    """
    Run the standard set of posterior predictive plots for single-outcome models.

    ``quantity`` says which outcome is being plotted, so every artefact below
    stops where that outcome's evidence stops. It is optional only so an
    engine that has not been told can still draw the full grid; the
    single-outcome pipeline always passes it.
    """
    max_age_months = (
        None if quantity is None
        else reporting_ages.max_age_for(context.model_config, quantity)
    )
    plotting.plot_posterior_predictive_count_distributions_by_query_age(
        X_query=context.model_samples.X_query,
        y_query=context.model_samples.y_query,
        n_trials=context.model_data.n_trials,
        ci_prob=context.reporting.ci_prob,
        output_dir=context.reporting.output_dir,
        filename="posterior_predictive_count_distributions",
        x_label=outcome_label,
        max_age_months=max_age_months,
    )

    plotting.plot_posterior_predictive_pmf(
        context.model_samples.X_query,
        context.model_samples.y_query,
        context.model_data.n_trials,
        output_dir=context.reporting.output_dir,
        filename="posterior_predictive_pmf",
        max_age_months=max_age_months,
    )

    plotting.plot_posterior_predictive_cdf(
        context.model_samples.X_query,
        context.model_samples.y_query,
        context.model_data.n_trials,
        output_dir=context.reporting.output_dir,
        filename="posterior_predictive_cdf",
        max_age_months=max_age_months,
    )

    plotting.plot_posterior_predictive_median_trend(
        context.model_samples.X_plot,
        context.model_samples.y_plot,
        context.model_samples.X_obs,
        context.model_samples.y_obs,
        output_dir=context.reporting.output_dir,
        filename="plot_posterior_predictive_median_trend",
        max_age_months=max_age_months,
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
        max_age_months=max_age_months,
    )

    plotting.plot_expected_learning_rate(
        context.model_samples.X_plot,
        context.model_samples.f_plot,
        n_trials=context.model_data.n_trials,
        ci_prob=context.reporting.ci_prob,
        output_dir=context.reporting.output_dir,
        filename="expected_learning_rate",
        max_age_months=max_age_months,
    )

    plotting.plot_expected_learning_rate(
        context.model_samples.X_plot,
        context.model_samples.f_plot,
        n_trials=context.model_data.n_trials,
        ci_prob=context.reporting.ci_prob,
        smooth=True,
        savgol_window_length=15,
        savgol_polyorder=3,
        smooth_intervals=True,
        output_dir=context.reporting.output_dir,
        filename="expected_learning_rate_smoothed",
        max_age_months=max_age_months,
    )

    plotting.plot_posterior_kappa(
        context.model_samples.X_plot,
        context.model_samples.kappa_plot,
        context.model_samples.X_query,
        context.model_samples.kappa_query,
        n_trials=context.model_data.n_trials,
        ci_prob=context.reporting.ci_prob,
        output_dir=context.reporting.output_dir,
        filename="posterior_kappa",
        max_age_months=max_age_months,
    )


def report(context: ModelFitContext):
    """
    Copy the model's Quarto source into its fitted output directory.

    The technical report's figure cache is deliberately refreshed only by
    ``scripts/sync_report_figures.py`` after all selected fits have passed the
    publication checks. This prevents a failed replacement fit from partially
    changing the report cache.
    """

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
            ("Quarto document", model_output_md_dest),
            (
                "Technical report cache",
                "Run scripts/sync_report_figures.py after validated fits complete",
            ),
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


class ConvergenceGateError(RuntimeError):
    """Raised when a reporting-quality fit fails convergence checks."""


def enforce_convergence_gate(
    gate_summary: dict,
    *,
    sampling_config_name: str,
    output_dir: str,
    model_id: str | None = None,
) -> list[str]:
    """Stop reporting pipelines whose reporting-quality posterior did not converge.

    Fails closed on the hard tier (the R-hat/ESS scan). Records the soft tier
    (divergences, energy BFMI) to :data:`CONVERGENCE_CAVEATS_FILENAME` and
    returns it, so a caveated fit stays reportable — the project's recorded
    decision — but can no longer be published as if it were clean. Returns the
    soft caveats (empty when the fit is clean).
    """
    if not is_reporting_quality_config(sampling_config_name):
        return []

    scan_failed = (
        gate_summary.get("max_rhat") is None
        or gate_summary.get("min_ess") is None
    )
    rhat_failing = gate_summary.get("rhat_failing") or []
    ess_failing = gate_summary.get("ess_failing") or []
    unassessable = gate_summary.get("unassessable_parameters") or []

    # A registered, narrowly-scoped exception can accept an R-hat-only failure
    # where the reported quantities converge and the failing parameter is a
    # nuisance direction. It is recorded into the diagnostics payload rather
    # than a marker file, because `convergence_caveats` recomputes from the
    # payload on disk -- so the exception reaches `check_fit`, the figure sync
    # and Appendix B by the same route as a divergence, and a fit carrying one
    # is publishable only through the `-with-caveats` purposes.
    accepted = accepted_rhat_exception(model_id, gate_summary)
    if accepted is not None:
        gate_summary[ACCEPTED_EXCEPTION_KEY] = {
            "parameters": list(accepted.parameters),
            "observed_max_rhat": gate_summary.get("max_rhat"),
            "ceiling_max_rhat": accepted.max_rhat,
            "reason": accepted.reason,
            "decided": accepted.decided,
        }
        if os.path.isfile(os.path.join(output_dir, DIAGNOSTICS_SUMMARY_FILENAME)):
            shared_diagnostics.amend_diagnostics_summary(
                output_dir, {ACCEPTED_EXCEPTION_KEY: gate_summary[ACCEPTED_EXCEPTION_KEY]}
            )
        rhat_failing = []
        scan_failed = False

    if scan_failed or rhat_failing or ess_failing or unassessable:
        if scan_failed:
            reason = "The R-hat/ESS convergence scan did not complete."
        elif unassessable and not (rhat_failing or ess_failing):
            reason = (
                f"The convergence gate could not assess R-hat/ESS for "
                f"{len(unassessable)} parameter(s): {', '.join(unassessable[:6])}."
            )
        else:
            reason = (
                f"The convergence gate found {len(rhat_failing)} R-hat failure(s) "
                f"and {len(ess_failing)} ESS failure(s)."
            )
        failure_path = os.path.join(output_dir, CONVERGENCE_FAILURE_FILENAME)
        with open(failure_path, "w", encoding="utf-8") as failure_file:
            failure_file.write(
                reason
                + "\nPosterior summaries, report generation, rendering, and upload must "
                "not proceed until the model is refitted successfully.\n"
            )
        raise ConvergenceGateError(
            reason
            + " Diagnostic artefacts were retained, but downstream publication artefacts "
            "were stopped."
        )

    # Hard tier passed. Record the soft tier durably — writing on every clean fit
    # too (by removing a stale file) so the marker's absence always means "checked
    # and clean", never "an older run never looked".
    caveats_path = os.path.join(output_dir, CONVERGENCE_CAVEATS_FILENAME)
    caveats = convergence_caveats(gate_summary)
    if caveats:
        with open(caveats_path, "w", encoding="utf-8") as caveats_file:
            caveats_file.write(
                "This fit cleared the hard convergence gate (R-hat and ESS) but "
                "carries sampling caveats:\n\n"
                + "".join(f"  - {caveat}\n" for caveat in caveats)
                + "\nThe fit remains reportable; publication as a clean fit is "
                "blocked. To publish it as what it is, use\n"
                "  sync_report_figures.py --allow-caveats\n"
                "which relaxes only this check and keeps every other publication "
                "requirement — reporting quality, rendered report, clean fit "
                "provenance, matching definition, sampling effort and raw-data "
                "fingerprint. --allow-provisional also proceeds, but it relaxes "
                "publication provenance more broadly for local development work, "
                "so it is not the publication path. Either way these caveats are "
                "written to the figure cache and reported with the numbers.\n"
            )
        console.print()
        for caveat in caveats:
            console.print(f"[bold yellow]⚠ convergence caveat: {caveat}[/bold yellow]")
    elif os.path.exists(caveats_path):
        os.remove(caveats_path)
    return caveats


def _report_diagnostic_warnings(gate_summary: dict) -> None:
    """Flag MCMC convergence issues recorded in the gate payload.

    ``gate_summary`` is the dict returned by
    ``shared_diagnostics.write_diagnostics_summary``, so the banner covers the
    same (unrounded) R-hat / ESS scan as the gate itself — every free
    parameter, element-wise — rather than the rounded scalar-only summary
    table.
    """
    thresholds = gate_summary.get("thresholds") or {}
    rhat_max = thresholds.get("rhat_max", shared_diagnostics.RHAT_MAX)
    ess_threshold = thresholds.get("ess_threshold", shared_diagnostics.ESS_THRESHOLD)
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


def build_univariate_analysis_frame(
    definition: UnivariateModelDefinition,
) -> tuple[pd.DataFrame, dict]:
    """The exact prepared frame the univariate engine fits, with no side effects.

    Split out of :func:`prepare_univariate_data` so fitted-output validation can
    recompute the frame (and its exact hash) without a fit context — see
    :mod:`vocab_growth.analysis_frames` (issue #266 finding 1).
    """
    y_col = definition.outcome.value
    df = vocab_data_utils.load_data(
        population=definition.population,
        # `study` and `subject_id` are provenance columns: the model itself sees
        # only age and outcome, but discarding them here left the fit manifest's
        # source counts empty and the frame unauditable (#234). They ride along
        # on the analysis frame; the model arrays below never read them.
        columns=["age", y_col, "study", "subject_id"],
        sample_fraction=definition.sample_fraction,
        random_seed=definition.random_seed,
        # TD language scope is part of the model graph; DS ignores it.
        languages=getattr(
            definition, "td_languages", vocab_data_utils.ENGLISH_LANGUAGES
        ),
    )
    analysis_df = df[["age", y_col, "study", "subject_id"]].dropna(
        subset=["age", y_col]
    )
    return analysis_df, {}


def prepare_univariate_data(
    context: ModelFitContext,
    definition: UnivariateModelDefinition,
):
    """Load and prepare data for a univariate model from its definition."""
    y_col = definition.outcome.value
    analysis_df, _ = build_univariate_analysis_frame(definition)

    desc = descriptive_stats.describe_all(analysis_df[["age", y_col]], alpha=0.05)

    n_children = int(analysis_df.groupby(["study", "subject_id"]).ngroups)
    key_value_table(
        "Data",
        [
            ("Population", definition.population.name),
            ("Outcome column", y_col),
            ("Rows after NA drop", len(analysis_df)),
            ("Children", n_children),
            ("Sources", int(analysis_df["study"].nunique())),
            ("Sample fraction", definition.sample_fraction),
        ],
    )
    dataframe_table(desc, title="Descriptive statistics")

    X_obs = np.asarray(analysis_df["age"], dtype=float).reshape(-1, 1)
    y_values = np.asarray(analysis_df[y_col], dtype=float)
    require_integral_counts(y_values, y_col)
    y_obs = y_values.astype(int)

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

    kappa_fields = _configure_kappa_priors(context, definition.kappa)
    partition_fields = _configure_variance_partition_priors(
        context, getattr(definition, "subject_variance_partition", None)
    )

    config = ModelConfiguration(
        slope_anchors=definition.slope_anchors,
        ell_months_range=definition.ell_months_range,
        p_slope_low_dist=p_slope_low_dist,
        p_slope_hi_dist=p_slope_hi_dist,
        ell_unit_dist=ell_unit_dist,
        eta_dist=eta_dist,
        n_plot=definition.n_plot,
        ages_query=definition.ages_query,
        report_max_age_understood=definition.report_max_age_understood,
        **kappa_fields,
        **partition_fields,
    )

    context.set_model_config(config)


def _configure_variance_partition_priors(context: ModelFitContext, vp) -> dict:
    """Build the scatter-budget priors, or nothing when the model does not use them.

    Plots both under their own names, as the other prior blocks do, so a report
    shows the quantities the model actually samples.
    """
    if vp is None:
        return {}
    total_dist = pz.LogNormal(mu=vp.total_mu, sigma=vp.total_sigma)
    _plot_and_print_dist(context, total_dist, "v_total_dist")
    share_dist = pz.Beta(alpha=vp.share_alpha, beta=vp.share_beta)
    _plot_and_print_dist(context, share_dist, "subject_variance_share_dist")
    return {
        "variance_partition_total_dist": total_dist,
        "variance_partition_share_dist": share_dist,
        "variance_partition_reference_proportion": vp.reference_proportion,
    }


def _configure_kappa_priors(context: ModelFitContext, kp, suffix: str = "") -> dict:
    """Build the dispersion priors for whichever kappa parameterisation `kp` is.

    Returns the configuration keyword arguments for that form, and plots each
    prior under its own name, so a report shows the parameters the model actually
    samples rather than a fixed three. ``suffix`` selects the outcome for the
    joint engines ("_u", "_s", "_sign"); it names both the configuration fields
    and the plotted priors.
    """
    if isinstance(kp, KappaAnchorPriorParams):
        kappa_min_dist = pz.LogNormal(mu=kp.kappa_min_mu, sigma=kp.kappa_min_sigma)
        _plot_and_print_dist(context, kappa_min_dist, f"kappa_min{suffix}_dist")

        excess_young_dist = pz.LogNormal(
            mu=kp.excess_young_mu, sigma=kp.excess_young_sigma
        )
        _plot_and_print_dist(
            context, excess_young_dist, f"kappa_excess_young{suffix}_dist"
        )

        excess_old_dist = pz.LogNormal(mu=kp.excess_old_mu, sigma=kp.excess_old_sigma)
        _plot_and_print_dist(
            context, excess_old_dist, f"kappa_excess_old{suffix}_dist"
        )

        return {
            f"kappa_anchored{suffix}": AnchoredKappaPriors(
                kappa_min_dist=kappa_min_dist,
                excess_young_dist=excess_young_dist,
                excess_old_dist=excess_old_dist,
                anchor_ages=tuple(float(age) for age in kp.anchor_ages),
            )
        }

    kappa_min_dist = pz.LogNormal(mu=kp.kappa_min_mu, sigma=kp.kappa_min_sigma)
    _plot_and_print_dist(context, kappa_min_dist, f"kappa_min{suffix}_dist")

    a_kappa_dist = pz.Normal(mu=kp.a_kappa_mu, sigma=kp.a_kappa_sigma)
    _plot_and_print_dist(context, a_kappa_dist, f"a_kappa{suffix}_dist")

    b_kappa_mag_dist = pz.HalfNormal(sigma=kp.b_kappa_mag_sigma)
    _plot_and_print_dist(context, b_kappa_mag_dist, f"b_kappa_mag{suffix}_dist")

    return {
        f"kappa_min{suffix}_dist": kappa_min_dist,
        f"a_kappa{suffix}_dist": a_kappa_dist,
        f"b_kappa_mag{suffix}_dist": b_kappa_mag_dist,
    }


# The exact-frame hash lives in ``vocab_growth.analysis_frames`` so validators
# can recompute it without importing the engines; kept under its historical
# name here for the manifest writer below.
_analysis_data_hash = analysis_frame_hash


def write_fit_manifest(context: ModelFitContext, definition) -> None:
    """Write a machine-readable provenance manifest for the prepared fit."""
    analysis_df = context.analysis_df
    source_counts = (
        {
            str(source): int(count)
            for source, count in analysis_df.groupby("study", dropna=False)
            .size()
            .items()
        }
        if "study" in analysis_df.columns
        else {}
    )
    outcome_counts = {
        column: int(analysis_df[column].notna().sum())
        for column in ("understood", "spoken", "signed")
        if column in analysis_df.columns
    }
    children = (
        int(analysis_df.groupby(["study", "subject_id"]).ngroups)
        if {"study", "subject_id"}.issubset(analysis_df.columns)
        else None
    )
    packages: dict[str, str] = {}
    direct_origins: dict[str, object] = {}
    for distribution in importlib_metadata.distributions():
        name = distribution.metadata.get("Name")
        if not name:
            continue
        packages[name] = distribution.version
        direct_url = distribution.read_text("direct_url.json")
        if direct_url:
            try:
                direct_origins[name] = json.loads(direct_url)
            except json.JSONDecodeError:
                direct_origins[name] = {"unparsed": direct_url.strip()}
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "model": {
            "model_id": definition.model_id,
            "config_name": definition.config_name,
            "definition": normalise_for_json(definition),
            # The same fields again, classified as graph-affecting,
            # data-affecting, reporting or identity, and versioned (issue #273).
            # Written *alongside* the raw dictionary rather than replacing it:
            # every fit on disk carries the raw form, several readers index it
            # directly, and the report layer reads its numbers out of it. A
            # reader that wants to know what kind of thing a field controls now
            # has it recorded rather than having to guess from the name.
            "definition_payload": fit_identity.semantic_payload(definition),
        },
        "sampling": {
            "configuration_name": context.sampling_config_name,
            "parameters": asdict(context.sampling),
        },
        "data": {
            "rows": len(analysis_df),
            "columns": list(analysis_df.columns),
            "n_trials": context.model_data.n_trials,
            "analysis_frame_hash": _analysis_data_hash(analysis_df),
            "source_data_hash": source_data_hash(local_env.DATA_DIR),
            "source_row_counts": source_counts,
            "observed_outcome_counts": outcome_counts,
            # None when the frame carries no subject identifiers (kept rather
            # than omitted so readers can tell "not recorded" from "zero").
            "children": children,
        },
        "code": git_metadata(local_env.ROOT_DIR),
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "packages": dict(sorted(packages.items(), key=lambda item: item[0].lower())),
            "direct_package_origins": dict(
                sorted(direct_origins.items(), key=lambda item: item[0].lower())
            ),
        },
    }
    manifest_path = os.path.join(context.reporting.output_dir, FIT_MANIFEST_FILENAME)
    write_json_atomic(manifest_path, manifest)


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
    is_reporting_quality_config(config)
    run_banner(definition.banner, subtitle=f"sampling config: {config}")

    env_info.report_environment_info()

    console.print()
    package_metadata.report_package_versions(PACKAGE_LIST)

    output_root = local_env.output_root()
    canonical_reporting = reporting.ReportingConfiguration(
        model_name=definition.model_id,
        config_name=definition.config_name,
        output_root_dir=output_root,
        ci_prob=0.89,
        interval_kind="eti",
    )
    staging_root = create_staging_root(output_root, canonical_reporting.model_label)
    context = ModelFitContext(
        reporting=reporting.ReportingConfiguration(
            model_name=definition.model_id,
            config_name=definition.config_name,
            output_root_dir=staging_root,
            ci_prob=0.89,
            interval_kind="eti",
        ),
        sampling=sampling.get_sampling_configuration(config),
        sampling_config_name=config,
    )

    os.makedirs(context.reporting.output_dir, exist_ok=True)
    write_fit_state(
        context.reporting.output_dir,
        "initialising",
        model_id=definition.model_id,
        config_name=definition.config_name,
        sampling_config_name=config,
    )

    timings = context.timings
    run_started = time.perf_counter()

    try:
        for stage_index, (name, fn) in enumerate(stages):
            write_fit_state(
                context.reporting.output_dir,
                "running",
                model_id=definition.model_id,
                config_name=definition.config_name,
                sampling_config_name=config,
                stage=name,
            )
            with section(name, timings=timings):
                fn(context)
                if stage_index == 0:
                    write_fit_manifest(context, definition)

        required_paths = [
            os.path.join(context.reporting.output_dir, FIT_MANIFEST_FILENAME),
            os.path.join(context.reporting.output_dir, "trace.nc"),
        ]
        missing = [path for path in required_paths if not os.path.isfile(path)]
        if missing:
            raise RuntimeError(
                "Fit pipeline reached finalisation without required artefact(s): "
                + ", ".join(os.path.basename(path) for path in missing)
            )
        write_fit_state(
            context.reporting.output_dir,
            "complete",
            model_id=definition.model_id,
            config_name=definition.config_name,
            sampling_config_name=config,
        )
        promote_staged_fit(
            context.reporting.output_dir,
            canonical_reporting.output_dir,
        )
        context.reporting.output_root_dir = output_root
    except BaseException as exc:
        if os.path.isdir(context.reporting.output_dir):
            write_fit_state(
                context.reporting.output_dir,
                "failed",
                model_id=definition.model_id,
                config_name=definition.config_name,
                sampling_config_name=config,
                error=exc,
            )
            retained = retain_failed_fit(context.reporting.output_dir, output_root)
            if retained is not None:
                console.print(f"[yellow]Failed fit retained for diagnosis: {retained}[/yellow]")
        raise
    finally:
        if os.path.isdir(staging_root):
            shutil.rmtree(staging_root)

    pipeline_summary(f"Pipeline summary — {context.reporting.model_label}", timings)
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
            (
                "Model definition and initialisation",
                lambda ctx: build_model(ctx, definition),
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
                lambda ctx: sample_posterior_predictive(ctx, definition),
            ),
            ("Posterior summary", posterior_summary),
            (
                "Plots",
                lambda ctx: run_standard_plots(
                    ctx,
                    outcome_label=outcome_label,
                    quantity=reporting_ages.quantity_for_outcome(definition.outcome),
                ),
            ),
            ("Report", report),
        ],
    )
