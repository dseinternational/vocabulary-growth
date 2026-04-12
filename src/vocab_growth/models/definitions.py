# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Model definitions for the vocabulary growth model family.

Each model is fully determined by its definition: population, outcome(s),
prior parameters, and data configuration. All procedural code (model building,
sampling, plotting, reporting) lives in common.py and common_bivariate.py.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum


class Population(Enum):
    """Study population."""

    DOWN_SYNDROME = "ds"
    TYPICALLY_DEVELOPING = "td"


class Outcome(Enum):
    """Vocabulary outcome variable."""

    SPOKEN = "spoken"
    UNDERSTOOD = "understood"


class ModelType(Enum):
    """Model structure type."""

    UNIVARIATE = "univariate"
    BIVARIATE = "bivariate"


# ============================================================
# Shared prior defaults (identical across all 6 models)
# ============================================================


@dataclass
class KappaPriorParams:
    """Parameters for the dispersion (kappa) prior distributions."""

    kappa_min_mu: float = math.log(5.0)
    """LogNormal mu for kappa_min."""
    kappa_min_sigma: float = 0.6
    """LogNormal sigma for kappa_min."""
    a_kappa_mu: float = math.log(8.0)
    """Normal mu for a_kappa."""
    a_kappa_sigma: float = 1.0
    """Normal sigma for a_kappa."""
    b_kappa_mag_sigma: float = 0.3
    """HalfNormal sigma for b_kappa_mag."""


# ============================================================
# Univariate model definition
# ============================================================


@dataclass
class UnivariateModelDefinition:
    """Complete definition for a single-outcome model (VG01-VG04)."""

    model_id: str
    """Model identifier, e.g. 'VG01'."""
    config_name: str
    """Configuration name, e.g. 'age-spoken-ds'."""
    banner: str
    """Banner text printed at fit start."""
    population: Population
    outcome: Outcome
    n_trials: int
    """Number of words on the vocabulary checklist."""
    slope_anchors: tuple[float, float]
    """Reference ages (months) for the slope parameterisation."""
    ages_query: list[int]
    """Ages (months) at which to query the posterior."""

    # -- Slope priors (the values that vary across models) --
    p_slope_low_alpha: float
    p_slope_low_beta: float
    p_slope_hi_alpha: float
    p_slope_hi_beta: float

    # -- TD-specific data parameters --
    sample_fraction: float = 1.0
    """Fraction of TD data to subsample (1.0 = no subsampling)."""
    random_seed: int = 47
    """Random seed for TD subsampling."""

    # -- Shared priors (same across all univariate models) --
    ell_unit_alpha: float = 3.0
    ell_unit_beta: float = 3.0
    eta_sigma: float = 0.4
    ell_months_range: tuple[int, int] = (6, 18)
    n_plot: int = 500
    kappa: KappaPriorParams = field(default_factory=KappaPriorParams)

    @property
    def model_type(self) -> ModelType:
        return ModelType.UNIVARIATE

    @property
    def outcome_label(self) -> str:
        return f"Words {self.outcome.value}"


# ============================================================
# Bivariate model definition
# ============================================================


@dataclass
class BivariateModelDefinition:
    """Complete definition for a joint understood+spoken model (VG05-VG06)."""

    model_id: str
    """Model identifier, e.g. 'VG05'."""
    config_name: str
    """Configuration name, e.g. 'age-understood-spoken-ds'."""
    banner: str
    """Banner text printed at fit start."""
    population: Population
    n_trials: int
    """Number of words on the vocabulary checklist."""
    slope_anchors: tuple[float, float]
    """Reference ages (months) for the slope parameterisation."""
    ages_query: list[int]
    """Ages (months) at which to query the posterior."""

    # -- Understood (U) trajectory slope priors --
    p_slope_low_u_alpha: float
    p_slope_low_u_beta: float
    p_slope_hi_u_alpha: float
    p_slope_hi_u_beta: float

    # -- Production ratio (q) slope priors --
    p_slope_low_q_alpha: float = 1.0
    p_slope_low_q_beta: float = 1.5
    p_slope_hi_q_alpha: float = 2.0
    p_slope_hi_q_beta: float = 1.2

    # -- TD-specific data parameters --
    sample_fraction: float = 1.0
    """Fraction of TD data to subsample (1.0 = no subsampling)."""
    random_seed: int = 47
    """Random seed for TD subsampling."""

    # -- Shared priors (same across all bivariate models) --
    ell_unit_u_alpha: float = 3.0
    ell_unit_u_beta: float = 3.0
    eta_u_sigma: float = 0.4
    ell_unit_q_alpha: float = 3.0
    ell_unit_q_beta: float = 3.0
    eta_q_sigma: float = 0.4
    ell_months_range: tuple[int, int] = (6, 18)
    n_plot: int = 500
    kappa_u: KappaPriorParams = field(default_factory=KappaPriorParams)
    kappa_s: KappaPriorParams = field(default_factory=KappaPriorParams)

    # -- Study-level random intercepts --
    include_study_re: bool = False
    """Whether to include study-level random intercepts."""
    tau_u_sigma: float = 0.5
    """HalfNormal scale for study intercept SD on understood (logit scale)."""
    tau_q_sigma: float = 0.5
    """HalfNormal scale for study intercept SD on production ratio (logit scale)."""

    @property
    def model_type(self) -> ModelType:
        return ModelType.BIVARIATE


# ============================================================
# Model instances
# ============================================================

VG01 = UnivariateModelDefinition(
    model_id="VG01",
    config_name="age-spoken-ds",
    banner="Fitting Model VG01: Influence of age on words spoken (A -> S)",
    population=Population.DOWN_SYNDROME,
    outcome=Outcome.SPOKEN,
    n_trials=800,
    slope_anchors=(24, 84),
    ages_query=[12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90],
    p_slope_low_alpha=1.0,
    p_slope_low_beta=15.0,
    p_slope_hi_alpha=1.1,
    p_slope_hi_beta=1.1,
)

VG02 = UnivariateModelDefinition(
    model_id="VG02",
    config_name="age-understood-ds",
    banner="Fitting Model VG02: Influence of age on words understood (A -> U)",
    population=Population.DOWN_SYNDROME,
    outcome=Outcome.UNDERSTOOD,
    n_trials=800,
    slope_anchors=(24, 84),
    ages_query=[12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90],
    p_slope_low_alpha=1.0,
    p_slope_low_beta=10.0,
    p_slope_hi_alpha=1.1,
    p_slope_hi_beta=1.1,
)

VG03 = UnivariateModelDefinition(
    model_id="VG03",
    config_name="age-spoken-td",
    banner="Fitting Model VG03: Influence of age on words spoken (A -> S)",
    population=Population.TYPICALLY_DEVELOPING,
    outcome=Outcome.SPOKEN,
    n_trials=690,
    slope_anchors=(12, 26),
    ages_query=[9, 12, 15, 18, 21, 24, 27, 30],
    p_slope_low_alpha=1.0,
    p_slope_low_beta=15.0,
    p_slope_hi_alpha=1.5,
    p_slope_hi_beta=1.1,
    sample_fraction=0.1,
)

VG04 = UnivariateModelDefinition(
    model_id="VG04",
    config_name="age-understood-td",
    banner="Fitting Model VG04: Influence of age on words understood (A -> U)",
    population=Population.TYPICALLY_DEVELOPING,
    outcome=Outcome.UNDERSTOOD,
    n_trials=690,
    slope_anchors=(12, 26),
    ages_query=[9, 12, 15, 18, 21, 24, 27, 30],
    p_slope_low_alpha=1.0,
    p_slope_low_beta=20.0,
    p_slope_hi_alpha=1.5,
    p_slope_hi_beta=1.1,
    sample_fraction=0.1,
)

VG05 = BivariateModelDefinition(
    model_id="VG05",
    config_name="age-understood-spoken-ds",
    banner=(
        "Fitting Model VG05: Joint model of words understood and spoken"
        " (A -> U, A -> S, U -> S)"
    ),
    population=Population.DOWN_SYNDROME,
    n_trials=800,
    slope_anchors=(24, 84),
    ages_query=[12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90],
    p_slope_low_u_alpha=1.0,
    p_slope_low_u_beta=10.0,
    p_slope_hi_u_alpha=1.1,
    p_slope_hi_u_beta=1.1,
)

VG06 = BivariateModelDefinition(
    model_id="VG06",
    config_name="age-understood-spoken-td",
    banner=(
        "Fitting Model VG06: Joint model of words understood and spoken"
        " (A -> U, A -> S, U -> S) - typically developing"
    ),
    population=Population.TYPICALLY_DEVELOPING,
    n_trials=690,
    slope_anchors=(12, 26),
    ages_query=[9, 12, 15, 18, 21, 24, 27, 30],
    p_slope_low_u_alpha=1.0,
    p_slope_low_u_beta=20.0,
    p_slope_hi_u_alpha=1.5,
    p_slope_hi_u_beta=1.1,
    sample_fraction=0.1,
)

VG07 = BivariateModelDefinition(
    model_id="VG07",
    config_name="age-understood-spoken-ds-re",
    banner=(
        "Fitting Model VG07: Joint model with study random intercepts"
        " (A -> U, A -> S, U -> S) - Down syndrome"
    ),
    population=Population.DOWN_SYNDROME,
    n_trials=800,
    slope_anchors=(24, 84),
    ages_query=[12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90],
    p_slope_low_u_alpha=1.0,
    p_slope_low_u_beta=10.0,
    p_slope_hi_u_alpha=1.1,
    p_slope_hi_u_beta=1.1,
    include_study_re=True,
    tau_u_sigma=0.5,
    tau_q_sigma=0.5,
)

MODEL_REGISTRY: dict[str, UnivariateModelDefinition | BivariateModelDefinition] = {
    "vg01": VG01,
    "vg02": VG02,
    "vg03": VG03,
    "vg04": VG04,
    "vg05": VG05,
    "vg06": VG06,
    "vg07": VG07,
}
