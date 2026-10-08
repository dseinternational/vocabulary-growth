# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Refit an engine to a synthetic analysis frame.

The engine's preparation stage is replaced by a frame loader. Its remaining
pipeline uses the requested fitted definition and sampling configuration, which
may differ from the generating definition in a controlled comparison. Recovery
outputs have separate names under ``models/`` and do not replace registered fits.
"""

from __future__ import annotations

import dataclasses
import os
from typing import Any

import numpy as np
import pandas as pd

from vocab_growth import environment as env
from vocab_growth.fit_artifacts import write_json_atomic
from vocab_growth.models.common import ModelFitContext, run_fit_pipeline
from vocab_growth.models.definitions import MODEL_REGISTRY
from vocab_growth.recovery.simulate import (
    PREPARE_STAGE_NAME,
    build_model_data,
    load_simulation,
    recovery_config_name,
    simulation_dir,
)
from vocab_growth.recovery.spec import recovery_target
from vocab_growth.reporting import key_value_table

RECOVERY_SOURCE_FILENAME = "recovery_source.json"


def make_recovery_definition(
    definition, replicate: int, *, truth_definition=None, truth_overrides=()
):
    """Copy the fitted definition with a recovery config name and banner.

    Priors, hyperparameters and structure remain unchanged. A distinct generating
    config name and truth settings contribute output identity and provenance.
    """
    if replicate < 1:
        raise ValueError("replicate is 1-based.")
    cross = (
        truth_definition is not None
        and truth_definition.config_name != definition.config_name
    )
    provenance = f"parameter recovery: replicate {replicate:02d}"
    if cross:
        provenance = (
            f"{provenance}, under {truth_definition.model_id} "
            f"[{truth_definition.config_name}]"
        )
    if truth_overrides:
        provenance = (
            f"{provenance}, truth set: "
            f"{', '.join(str(item) for item in truth_overrides)}"
        )
    return dataclasses.replace(
        definition,
        config_name=recovery_config_name(
            definition,
            replicate,
            truth_definition=truth_definition,
            truth_overrides=truth_overrides,
        ),
        banner=f"{definition.banner} [{provenance}]",
    )


def _validate_frame(frame: pd.DataFrame, definition, columns: list[str]) -> None:
    """Require age and check finite numeric outcome values against checklist bounds.

    This range check is not a complete validation of the generating likelihood;
    it does not check integrality, missingness, nesting or cross-tab coherence.
    """
    if "age" not in frame.columns:
        raise ValueError("Synthetic frame has no 'age' column.")
    n_trials = definition.n_trials
    for column in columns:
        if column not in frame.columns:
            continue
        values = pd.to_numeric(frame[column], errors="coerce").dropna().to_numpy()
        if values.size == 0:
            continue
        if np.any(values < 0) or np.any(values > n_trials):
            raise ValueError(
                f"Synthetic {column} outside [0, {n_trials}]; the simulation and this "
                "definition disagree."
            )


def _loader_stage(frame: pd.DataFrame, definition, record: dict[str, Any]):
    """Build the stage that injects a simulated frame in place of data preparation."""

    def load_synthetic_data(context: ModelFitContext) -> None:
        simulated = record.get("simulation", {})
        _validate_frame(frame, definition, list(simulated.get("simulated_columns", [])))
        context.set_model_data(build_model_data(frame, definition), frame.copy())
        key_value_table(
            "Simulated data (parameter recovery)",
            [
                ("Rows", len(frame)),
                ("Simulated columns", ", ".join(simulated.get("simulated_columns", []))),
                ("Truth source", simulated.get("truth_source", "unknown")),
                (
                    "Truth draw",
                    f"chain {simulated.get('truth_chain')}, draw {simulated.get('truth_draw')}",
                ),
                ("Replicate", simulated.get("replicate")),
                ("Coherence checks", len(simulated.get("coherence_checks", {}))),
            ],
        )
        # Written inside the staged output so it is promoted with the fit: a
        # recovery fit must carry the identity of the truth it is chasing.
        write_json_atomic(
            os.path.join(context.reporting.output_dir, RECOVERY_SOURCE_FILENAME),
            {
                "schema_version": 1,
                "parameter_recovery": True,
                "source_model": record.get("model", {}),
                "simulation": simulated,
            },
        )

    return load_synthetic_data


def fit_recovery_replicate(
    model_key: str,
    config: str,
    *,
    replicate: int = 1,
    output_root: str | None = None,
    definition=None,
    fit_definition=None,
    truth_overrides=(),
) -> ModelFitContext:
    """Refit a requested definition to one replicate's synthetic frame.

    ``definition`` identifies the generating model or sensitivity variant and
    validates the stored simulation. ``fit_definition`` identifies the model
    fitted to those data, defaulting to the generating definition. Engine stages
    resolve from the base registry key; only preparation is substituted.

    Keeping simulated data fixed while changing the fitted definition supports
    comparisons of fitting choices without also changing the generating truth.
    """
    target = recovery_target(model_key)
    definition = MODEL_REGISTRY[model_key] if definition is None else definition
    fit_definition = definition if fit_definition is None else fit_definition
    truth_overrides = tuple(truth_overrides)
    root = output_root if output_root is not None else env.output_root()
    directory = simulation_dir(
        definition, replicate, root, truth_overrides=truth_overrides
    )
    if not os.path.isdir(directory):
        raise FileNotFoundError(
            f"No simulated data at {directory}. Run the simulate step for "
            f"{model_key} replicate {replicate} first."
        )
    frame, _truth, record = load_simulation(directory, expected_definition=definition)

    recovery_definition = make_recovery_definition(
        fit_definition,
        replicate,
        truth_definition=definition,
        truth_overrides=truth_overrides,
    )
    stages = target.resolve_stages(recovery_definition)
    if stages[0][0] != PREPARE_STAGE_NAME:
        raise RuntimeError(
            f"Expected the first engine stage to be {PREPARE_STAGE_NAME!r}, "
            f"found {stages[0][0]!r}."
        )
    stages[0] = (
        "Load simulated data",
        _loader_stage(frame, recovery_definition, record),
    )
    return run_fit_pipeline(config, recovery_definition, stages=stages)


def recovery_fit_dir(
    model_key: str,
    replicate: int,
    output_root: str | None = None,
    definition=None,
    truth_definition=None,
    truth_overrides=(),
) -> str:
    """Directory a recovery replicate's fit is promoted to.

    ``definition`` is the model that was fitted and ``truth_definition`` the one
    the data came from, matching :func:`fit_recovery_replicate`'s two arguments.
    """
    import dse_research_utils.statistics.models.reporting as model_reporting

    definition = MODEL_REGISTRY[model_key] if definition is None else definition
    root = output_root if output_root is not None else env.output_root()
    return model_reporting.ReportingConfiguration(
        model_name=definition.model_id,
        config_name=recovery_config_name(
            definition,
            replicate,
            truth_definition=truth_definition,
            truth_overrides=truth_overrides,
        ),
        output_root_dir=root,
        ci_prob=0.89,
        interval_kind="eti",
    ).output_dir
