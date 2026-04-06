# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Shared dataclasses for the vocabulary growth model family.
"""

from dataclasses import dataclass, field
from typing import Generic, TypeVar

import dse_research_utils.environment.info as env_info
import dse_research_utils.statistics.models.data as model_data
import dse_research_utils.statistics.models.reporting as reporting
import dse_research_utils.statistics.models.sampling as sampling
import numpy as np
import pandas as pd
import pymc as pm
from arviz import ELPDData, InferenceData
from matplotlib.figure import Figure
from preliz.distributions.distributions import Continuous


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
    _model_data: model_data.BinomialModelData | None = None
    _analysis_df: pd.DataFrame | None = None
    _model_config: C | None = None
    _model: pm.Model | None = None
    _model_variables: dict | None = None
    _prior_samples: InferenceData | None = None
    _trace: InferenceData | None = None
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

    def set_prior_samples(self, prior_samples: InferenceData):
        self._prior_samples = prior_samples

    @property
    def prior_samples(self) -> InferenceData:
        if self._prior_samples is None:
            raise ValueError("Prior samples have not been set in the context.")
        return self._prior_samples

    def set_trace(self, trace: InferenceData):
        self._trace = trace

    @property
    def trace(self) -> InferenceData:
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
