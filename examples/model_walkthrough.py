# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

# %% [markdown]
# > Drafted by an LLM-based AI tool (OpenAI Codex/GPT-6).
#
# # Build a vocabulary model step by step
#
# Read docs/tutorials/model-code-walkthrough.md alongside this file.
# Run `uv run python examples/model_walkthrough.py` from the repository root.
# These percent-format cells can also be opened as a Jupytext notebook.
# All counts below are invented. Building a graph does not fit a posterior.

# %%
import importlib
from dataclasses import replace
from pathlib import Path

import dse_research_utils.statistics.models.data as model_data
import dse_research_utils.statistics.models.reporting as reporting
import dse_research_utils.statistics.models.sampling as sampling
import numpy as np
import pandas as pd
import pymc as pm
from scipy.stats import betabinom, binom

from vocab_growth.models.build_utils import require_valid_counts
from vocab_growth.models.catalogue import get
from vocab_growth.models.common import ModelFitContext
from vocab_growth.models.composition import composition_probabilities
from vocab_growth.models.definitions import (
    VG01,
    VG05,
    VG07,
    VG10,
    VG19,
    VG20,
    validate_model_definition,
)
from vocab_growth.models.observation_arrays import prepare_bivariate_observations
from vocab_growth.models.subject_effects import resolve


def small_frame() -> pd.DataFrame:
    """Twelve administrations, six children and three studies; ages in months."""
    child = np.repeat(np.arange(6), 2)
    return pd.DataFrame(
        {
            "age": [24, 36, 24, 36, 30, 42, 30, 42, 36, 48, 36, 48],
            "understood": [60, 180, 90, 240, 90, 270, 140, 390, 230, 500, 200, 440],
            "spoken": [10, 45, 20, 80, 15, 70, 30, 130, 60, 260, 55, 180],
            "subject_id": child.astype(str),
            "subject_code": child,
            "study": np.array(["study_a", "study_b", "study_c"])[child // 2],
            "study_code": child // 2,
        }
    )


def build_example(definition, frame, *, output_dir=Path("output/teaching")):
    """Run the real prior configuration and graph builder without report files.

    The example supplies a prepared frame directly. A study fit instead runs
    the catalogue's data-preparation stage, with its source and exclusion rules.
    """
    validate_model_definition(definition)
    engine = get(definition.model_id.lower()).engine
    outcome = getattr(definition, "outcome", None)
    values = frame[outcome.value if outcome is not None else "understood"]
    require_valid_counts(
        values.dropna().to_numpy(), "example counts", definition.n_trials
    )
    if outcome is not None and values.isna().any():
        raise ValueError(
            "Remove rows with missing outcomes for a single-outcome example."
        )
    context = ModelFitContext(
        reporting=reporting.ReportingConfiguration(
            model_name=definition.model_id,
            config_name="teaching",
            output_root_dir=str(output_dir),
            ci_prob=0.89,
            interval_kind="eti",
        ),
        sampling=sampling.SamplingConfiguration(
            draws=8,
            tune=8,
            chains=1,
            cores=1,
            target_accept=0.8,
            random_seed=42,
        ),
        report_build=False,
    )
    context.set_model_data(
        model_data.BinomialModelData(
            X_obs=frame["age"].to_numpy(dtype=float).reshape(-1, 1),
            # Multi-outcome builders read missingness from the frame, not this placeholder.
            y_obs=values.fillna(0).to_numpy(dtype=int),
            n_trials=definition.n_trials,
        ),
        frame.copy(),
    )
    engine.resolve("priors")(context, definition)
    details = importlib.import_module(engine.module).build_model_graph(
        context, definition
    )
    return context, details


# %% [markdown]
# ## Counts and concentration
#
# For a 100-word checklist and mean proportion 0.2, all these distributions
# have mean count 20. Smaller concentration permits greater count variation.


# %%
def count_variation() -> pd.DataFrame:
    n, p = 100, 0.2
    rows = [
        {
            "distribution": "Binomial",
            "mean": binom.mean(n, p),
            "variance": binom.var(n, p),
        }
    ]
    for kappa in (2, 20, 200):
        mean, variance = betabinom.stats(n, p * kappa, (1 - p) * kappa, moments="mv")
        rows.append(
            {
                "distribution": f"Beta-Binomial, kappa={kappa}",
                "mean": float(mean),
                "variance": float(variance),
            }
        )
    return pd.DataFrame(rows)


# %% [markdown]
# ## Use the study's builders
#
# The example below retains each registered model's 810-word reference.
# VG01 models understood counts. VG05 adds speech conditional on understood.
# VG07 adds study offsets; VG10 adds child offsets; VG20 correlates them.
# `details.tables` contains the build settings for inspection without printing.


# %%
def run_walkthrough():
    frame = small_frame()
    contexts = {}
    for definition in (VG01, VG05, VG07, VG10, VG20):
        context, _ = build_example(replace(definition, n_plot=24), frame)
        contexts[definition.model_id] = context

    # This draws from prior assumptions, before the invented counts update them.
    with contexts["VG01"].model:
        prior = pm.sample_prior_predictive(
            draws=8,
            var_names=["p_query", "kappa_query"],
            random_seed=42,
        )

    # Missing U removes that comprehension likelihood term. The default spoken
    # treatment retains S through a full-inventory marginal likelihood.
    missing_parent = frame.astype({"understood": float})
    missing_parent.loc[0, "understood"] = np.nan
    observations = prepare_bivariate_observations(
        missing_parent,
        VG10,
        n_trials=VG10.n_trials,
        use_subject_codes=True,
    )

    one_study = frame.assign(study="one_study", study_code=0)
    one_study_context, _ = build_example(replace(VG20, n_plot=24), one_study)
    zero_reference = replace(VG19, n_plot=24, subject_slope_ref_age_months=0.0)
    slope_context, _ = build_example(zero_reference, frame)

    # Signed and spoken shares overlap. These are four probabilities within U,
    # ordered neither, sign only, speech only, both. Independence is not a bound.
    cells = composition_probabilities(
        np.array([0.4]), np.array([0.5]), np.array([1 / 6])
    ).eval()[0]
    return {
        "contexts": contexts,
        "prior": prior,
        "missing_parent": observations,
        "one_study": one_study_context,
        "slope": slope_context,
        "reference_age": resolve(zero_reference).slope_ref_age_months,
        "cells": cells,
    }


# %%
if __name__ == "__main__":
    print(count_variation().round(2).to_string(index=False))
    results = run_walkthrough()
    for name, context in results["contexts"].items():
        print(
            f"{name}: {len(context.model.free_RVs)} sampled variable arrays; no posterior fit performed."
        )
    print(
        f"Missing-parent speech fallback rows: {results['missing_parent'].spoken_spec.n_marginal}"
    )
    print(f"One-study offset: {results['one_study'].model['delta_u'].eval()}")
    print(f"Child-slope reference age: {results['reference_age']:g} months")
    print(f"Four cell probabilities: {results['cells'].round(3)}")
