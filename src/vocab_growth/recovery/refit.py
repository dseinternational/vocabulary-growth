# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Refit a model to its own simulated data (issue #163).

The refit must be the *same* fit the study publishes, or the check proves
nothing about the study. It therefore runs the engine's own pipeline — the same
priors, the same build, the same sampler settings, the same diagnostics gate —
with exactly one stage substituted: data preparation is replaced by a loader that
injects the simulated analysis frame produced by
:mod:`vocab_growth.recovery.simulate`.

Output lands in its own ``models/<model_id>-<config_name>-recovery-rNN/``
directory, so a recovery fit can never overwrite or be mistaken for a model of
record (``sync_report_figures`` and ``check_fit`` both key off the registered
config names and skip these labels).
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


def make_recovery_definition(definition, replicate: int):
    """Return a copy of ``definition`` whose output lands in a recovery directory.

    Only the identity fields change. Every prior, every hyperparameter and every
    structural flag is the model of record's, because the question being asked is
    whether *this* model recovers its own parameters.
    """
    if replicate < 1:
        raise ValueError("replicate is 1-based.")
    return dataclasses.replace(
        definition,
        config_name=recovery_config_name(definition, replicate),
        banner=f"{definition.banner} [parameter recovery: replicate {replicate:02d}]",
    )


def _validate_frame(frame: pd.DataFrame, definition, columns: list[str]) -> None:
    """Reject a synthetic frame the engine could not have produced."""
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
) -> ModelFitContext:
    """Refit ``model_key`` to replicate ``replicate``'s simulated data.

    ``definition`` overrides the model of record, for recovering a registered
    *sensitivity variant* — Proposal A1 is the first, and its whole claim is a
    structural one, so it needs recovery under its own structure rather than the
    record's. The engine plumbing still resolves from ``model_key``: a variant
    shares its base model's engine by construction, and reading the spec off the
    variant would let a mis-registered override quietly select a different one.
    """
    target = recovery_target(model_key)
    definition = MODEL_REGISTRY[model_key] if definition is None else definition
    root = output_root if output_root is not None else env.output_root()
    directory = simulation_dir(definition, replicate, root)
    if not os.path.isdir(directory):
        raise FileNotFoundError(
            f"No simulated data at {directory}. Run the simulate step for "
            f"{model_key} replicate {replicate} first."
        )
    frame, _truth, record = load_simulation(directory)

    recovery_definition = make_recovery_definition(definition, replicate)
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
) -> str:
    """Directory a recovery replicate's fit is promoted to."""
    import dse_research_utils.statistics.models.reporting as model_reporting

    definition = MODEL_REGISTRY[model_key] if definition is None else definition
    root = output_root if output_root is not None else env.output_root()
    return model_reporting.ReportingConfiguration(
        model_name=definition.model_id,
        config_name=recovery_config_name(definition, replicate),
        output_root_dir=root,
        ci_prob=0.89,
        interval_kind="eti",
    ).output_dir
