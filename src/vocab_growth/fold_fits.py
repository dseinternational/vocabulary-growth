# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Shared fitting and diagnostics for cross-validation folds.

``kfold_loso.py`` holds out whole children; ``wave_forward_score.py`` holds out
later administration waves. Both use a prepared frame with a ``holdout`` column,
the definition's registered engine, observation-level deterministics for scoring
and the full fit-pipeline diagnostics scan.

Callers choose held-out rows, scoring units and how to handle failed folds.
"""

from __future__ import annotations

import os
import shutil
from typing import Any

import dse_research_utils.statistics.diagnostics as shared_diagnostics
import dse_research_utils.statistics.models.data as model_data
import dse_research_utils.statistics.models.reporting as reporting
import dse_research_utils.statistics.models.sampling as sampling
import numpy as np
import pandas as pd

from vocab_growth.models.build_utils import require_valid_counts
from vocab_growth.models.catalogue import engine_for_definition
from vocab_growth.models.common import ModelFitContext, diagnostics_var_names, sample
from vocab_growth.models.definitions import UnivariateModelDefinition


def fold_gate_fields(gate: dict) -> dict:
    """A fold's convergence verdict, flattened for a table row.

    ``gate`` is the payload :func:`write_diagnostics_summary` returns. The energy
    check is nested under ``checks``, not at the top level.
    """
    checks = gate.get("checks") or {}
    return {
        "passed": bool(gate.get("passed")),
        "max_rhat": gate.get("max_rhat"),
        "min_ess": gate.get("min_ess"),
        "divergences": gate.get("divergences"),
        "bfmi_ok": checks.get("bfmi"),
    }


def fit_holdout_fold(
    definition,
    analysis_df_with_holdout: pd.DataFrame,
    sampling_cfg: sampling.SamplingConfiguration,
    *,
    label: str,
    tmp_root: str,
    name_prefix: str,
) -> tuple[Any, dict]:
    """Build, sample and diagnose a fold from a prepared, marked frame.

    ``analysis_df_with_holdout`` is a prepared frame carrying a boolean
    ``holdout`` column: those rows leave the likelihood but stay in ``obs_id``
    space, so the observation-level deterministics are still computed at their
    ages and a caller can read a held-out row's predictive density straight off
    the trace.

    Returns ``(trace, gate)``. The gate payload has to come back to the caller
    rather than being read from the JSON it also writes: the fold's directory is
    deleted when the next fold reuses it, so that file is transient.

    ``enforce_convergence_gate`` is deliberately not called. It is a no-op below
    the reporting tier, and a cross-validation run wants a failed fold recorded
    and flagged rather than the run aborted -- but that is the *caller's* choice,
    and both callers make it explicitly.

    The prior and build stages come from the definition's own engine
    (:func:`~vocab_growth.models.catalogue.engine_for_definition`), so a fold of
    a joint model is built by the joint builder. The rest is engine-independent:
    ``BinomialModelData`` is assembled the same way by every engine's prepare
    stage, ``sample`` is one shared function, and the diagnostics scan reads the
    model it is given.
    """
    context = build_holdout_fold_context(
        definition,
        analysis_df_with_holdout,
        sampling_cfg,
        label=label,
        tmp_root=tmp_root,
        name_prefix=name_prefix,
    )
    reporting_cfg = context.reporting
    # Scoring needs per-row probabilities, dispersions and composition
    # probabilities. Retain these arrays to avoid a separate recomputation.
    sample(context, store_observation_deterministics=True)

    # Use the fit pipeline's scan, including every free-variable element.
    _summary_names, gate_var_names = diagnostics_var_names(context.model)
    gate = shared_diagnostics.write_diagnostics_summary(
        context.trace, reporting_cfg.output_dir, var_names=gate_var_names
    )
    return context.trace, gate


def build_holdout_fold_context(
    definition,
    analysis_df_with_holdout: pd.DataFrame,
    sampling_cfg: sampling.SamplingConfiguration,
    *,
    label: str,
    tmp_root: str,
    name_prefix: str,
) -> ModelFitContext:
    """The prior and build stages of :func:`fit_holdout_fold`, without sampling.

    Tests can inspect the fold's likelihood rows and deterministic values
    without sampling.
    """
    engine = engine_for_definition(definition)
    # A univariate definition names its single outcome; every other engine's
    # `BinomialModelData` carries comprehension, as their prepare stages build it.
    count_col = (
        definition.outcome.value
        if isinstance(definition, UnivariateModelDefinition)
        else "understood"
    )
    has_u = analysis_df_with_holdout[count_col].notna().to_numpy()
    # Validate before casting, as engine preparation does. NumPy would silently
    # truncate fractional counts toward zero.
    require_valid_counts(
        np.asarray(analysis_df_with_holdout.loc[has_u, count_col], dtype=float),
        count_col,
        definition.n_trials,
    )
    bmd = model_data.BinomialModelData(
        X_obs=np.asarray(analysis_df_with_holdout["age"], dtype=float).reshape(-1, 1),
        y_obs=np.where(
            has_u, analysis_df_with_holdout[count_col].fillna(0).astype(int), 0
        ).astype(int),
        n_trials=definition.n_trials,
    )

    reporting_cfg = reporting.ReportingConfiguration(
        model_name=f"{name_prefix}-{label}",
        config_name=definition.config_name,
        output_root_dir=tmp_root,
        ci_prob=0.89,
        interval_kind="eti",
    )
    if os.path.exists(reporting_cfg.output_dir):
        shutil.rmtree(reporting_cfg.output_dir)
    os.makedirs(reporting_cfg.output_dir, exist_ok=True)

    context: ModelFitContext = ModelFitContext(
        reporting=reporting_cfg, sampling=sampling_cfg
    )
    context.set_model_data(bmd, analysis_df_with_holdout)
    engine.resolve("priors")(context, definition)
    engine.resolve("build")(context, definition)
    return context
