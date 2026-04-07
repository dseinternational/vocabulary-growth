# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Model VG01: Influence of age on words spoken (A → S) - children with Down syndrome
"""

import dse_research_utils.plot.distributions as plot_dist
import dse_research_utils.statistics.descriptive as descriptive_stats
import dse_research_utils.statistics.models.data as model_data
import numpy as np
import pandas as pd
import preliz as pz
from rich import print
from rich.pretty import pprint

import vocab_growth.data_utils as vocab_data_utils
from vocab_growth.models.common import (
    ModelConfiguration,
    ModelFitContext,
    fit_single_outcome_model,
)


def prepare_model_data(
    x_col: str = "age",
    y_col: str = "spoken",
    max_age_months: int | None = None,
    n_trials: int = 800,
) -> tuple[model_data.BinomialModelData, pd.DataFrame, pd.DataFrame]:

    vocab_df = vocab_data_utils.load_combined_data(max_age_months=max_age_months)
    analysis_df = vocab_df[[x_col, y_col]].dropna()

    desc = descriptive_stats.describe_all(analysis_df, alpha=0.05)

    print(
        "\n[green]------------------------------------------------------------[/green]"
    )
    print("[bold green]Descriptive statistics[/bold green]")
    print("[green]------------------------------------------------------------[/green]")

    pprint(desc)

    X_obs = np.asarray(analysis_df[x_col], dtype=float).reshape(-1, 1)  # (n, 1)
    y_obs = np.asarray(analysis_df[y_col], dtype=int)  # (n,)

    return (
        model_data.BinomialModelData(X_obs=X_obs, y_obs=y_obs, n_trials=n_trials),
        analysis_df,
        desc,
    )


def configure_model(context: ModelFitContext):
    """
    Configure priors and hyperparameters.
    """
    print(
        "\n[green]------------------------------------------------------------[/green]"
    )
    print("[bold green]Priors and hyperparameters[/bold green]")
    print("[green]------------------------------------------------------------[/green]")
    print()

    # Length scale
    ell_unit_dist = pz.Beta(alpha=3.0, beta=3.0)
    context.plots["ell_unit_dist"] = plot_dist.plot_distribution(
        ell_unit_dist, context.reporting.output_dir, "ell_unit_dist"
    )
    print(
        f"[bold yellow]ell_unit_dist:[/bold yellow] {ell_unit_dist.summary(mass=context.reporting.hdi)}"
    )

    # Amplitude
    eta_dist = pz.HalfNormal(sigma=0.4)
    context.plots["eta_dist"] = plot_dist.plot_distribution(
        eta_dist, context.reporting.output_dir, "eta_dist"
    )
    print(
        f"[bold yellow]eta_dist:[/bold yellow]: {eta_dist.summary(mass=context.reporting.hdi)}"
    )

    # Slope
    slope_age_low = 24
    slope_age_hi = 84

    p_slope_low_dist = pz.Beta(alpha=1.0, beta=15)
    context.plots["p_slope_low_dist"] = plot_dist.plot_distribution(
        p_slope_low_dist, context.reporting.output_dir, "p_slope_low_dist"
    )
    print(
        f"[bold yellow]p_slope_low_dist:[/bold yellow] {p_slope_low_dist.summary(mass=context.reporting.hdi)}"
    )

    p_slope_hi_dist = pz.Beta(alpha=1.1, beta=1.1)
    context.plots["p_slope_hi_dist"] = plot_dist.plot_distribution(
        p_slope_hi_dist, context.reporting.output_dir, "p_slope_hi_dist"
    )
    print(
        f"[bold yellow]p_slope_hi_dist:[/bold yellow] {p_slope_hi_dist.summary(mass=context.reporting.hdi)}"
    )

    # Dispersion / overdispersion

    kappa_min_dist = pz.LogNormal(mu=np.log(5.0), sigma=0.6)
    context.plots["kappa_min_dist"] = plot_dist.plot_distribution(
        kappa_min_dist, context.reporting.output_dir, "kappa_min_dist"
    )
    print(
        f"[bold yellow]kappa_min_dist:[/bold yellow] {kappa_min_dist.summary(mass=context.reporting.hdi)}"
    )

    a_kappa_dist = pz.Normal(mu=np.log(8.0), sigma=1.0)
    context.plots["a_kappa_dist"] = plot_dist.plot_distribution(
        a_kappa_dist, context.reporting.output_dir, "a_kappa_dist"
    )
    print(
        f"[bold yellow]a_kappa_dist:[/bold yellow] {a_kappa_dist.summary(mass=context.reporting.hdi)}"
    )

    b_kappa_mag_dist = pz.HalfNormal(sigma=0.3)
    context.plots["b_kappa_mag_dist"] = plot_dist.plot_distribution(
        b_kappa_mag_dist, context.reporting.output_dir, "b_kappa_mag_dist"
    )
    print(
        f"[bold yellow]b_kappa_mag_dist:[/bold yellow] {b_kappa_mag_dist.summary(mass=context.reporting.hdi)}"
    )

    # ------------------------------------------------------------
    # Model definition and initialisation
    # ------------------------------------------------------------

    config = ModelConfiguration(
        slope_anchors=(slope_age_low, slope_age_hi),
        ell_months_range=(2, 12),
        p_slope_low_dist=p_slope_low_dist,
        p_slope_hi_dist=p_slope_hi_dist,
        ell_unit_dist=ell_unit_dist,
        eta_dist=eta_dist,
        kappa_min_dist=kappa_min_dist,
        a_kappa_dist=a_kappa_dist,
        b_kappa_mag_dist=b_kappa_mag_dist,
        n_plot=500,
        ages_query=[12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90],
    )

    context.set_model_config(config)


def fit(config: str) -> ModelFitContext:
    return fit_single_outcome_model(
        config,
        model_name="VG01",
        config_name="age-spoken-ds",
        banner="Fitting Model VG01: Influence of age on words spoken (A -> S)",
        prepare_model_data_fn=prepare_model_data,
        y_col="spoken",
        outcome_label="Words spoken",
        configure_model_fn=configure_model,
    )
