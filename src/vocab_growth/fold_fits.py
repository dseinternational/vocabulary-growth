# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""One way to fit a cross-validation fold, shared by the scripts that need one.

``kfold_loso.py`` and ``wave_forward_score.py`` hold different things out --
whole children against a child's later administration waves -- and score
different units, but the fit between those two decisions is the same one:
validate the counts, build the model on a frame carrying a ``holdout`` column,
sample it with the observation-level deterministics stored, and run the canonical
diagnostics scan over every free variable element-wise.

It lives here rather than in either script because the second copy was made by
hand and was wrong within an hour: ``fold_gate_fields`` reads the energy verdict
from ``gate["checks"]["bfmi"]``, and the copy read ``gate["bfmi_ok"]``, which is
absent -- so every fold in the new script reported a passing energy check
regardless of what the sampler found. Nothing in either script's output would
have shown it.

Neither caller's *policy* moves here. Which rows are held out, which are scored,
whether a failed fold aborts the run or is recorded and flagged -- those differ
between the two and belong with the question each is asking. What is shared is
the mechanics.
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
from vocab_growth.models.common import ModelFitContext, diagnostics_var_names
from vocab_growth.models.common_bivariate import configure_bivariate_priors, sample
from vocab_growth.models.common_bivariate_re import build_model_re


def fold_gate_fields(gate: dict) -> dict:
    """A fold's convergence verdict, flattened for a table row.

    ``gate`` is the payload :func:`write_diagnostics_summary` returns. The energy
    check is nested under ``checks``, not at the top level, which is the detail a
    hand copy of this function got wrong.
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
    """prepare -> priors -> build -> sample -> diagnostics on a marked frame.

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
    """
    has_u = analysis_df_with_holdout["understood"].notna().to_numpy()
    # The engines' own prepare stage validates before the cast, because NumPy
    # truncates toward zero silently and a fold path builds its
    # `BinomialModelData` here rather than going through the engine (#233).
    require_valid_counts(
        np.asarray(analysis_df_with_holdout.loc[has_u, "understood"], dtype=float),
        "understood",
        definition.n_trials,
    )
    bmd = model_data.BinomialModelData(
        X_obs=np.asarray(analysis_df_with_holdout["age"], dtype=float).reshape(-1, 1),
        y_obs=np.where(
            has_u, analysis_df_with_holdout["understood"].fillna(0).astype(int), 0
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
    configure_bivariate_priors(context, definition)
    build_model_re(context, definition)
    # Both callers read `p_u_obs` / `p_s_obs` / `q_obs` / `kappa_*_obs` at every
    # draw to score held-out rows, and the sampler otherwise no longer stores
    # them (`fit_artifacts.sampled_variable_names`). Storing them costs the same
    # memory as recomputing them afterwards and saves the second pass.
    sample(context, store_observation_deterministics=True)

    # The scan's var_names are built exactly as the fit pipeline's diagnostics
    # stage builds them: the scalar summary set plus every free RV element-wise,
    # so the study and subject intercepts and the HSGP coefficients are screened
    # too rather than only the scalars.
    _summary_names, gate_var_names = diagnostics_var_names(context.model)
    gate = shared_diagnostics.write_diagnostics_summary(
        context.trace, reporting_cfg.output_dir, var_names=gate_var_names
    )
    return context.trace, gate
