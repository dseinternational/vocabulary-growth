# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Resume complete recovery stages without changing their validation policy."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import dse_research_utils.statistics.models.sampling as sampling

from vocab_growth import environment as env
from vocab_growth.analysis_frames import (
    analysis_frame_hash,
    expected_analysis_frame_hash,
)
from vocab_growth.fit_artifacts import (
    fit_validation_kwargs,
    git_metadata,
    normalise_for_json,
    source_data_hash,
    validate_fit_output,
)
from vocab_growth.fit_consumers import require_current_fit
from vocab_growth.models.common import is_reporting_quality_config
from vocab_growth.recovery.refit import (
    fit_recovery_replicate,
    make_recovery_definition,
    recovery_fit_dir,
)
from vocab_growth.recovery.simulate import (
    SIMULATION_FILENAME,
    SYNTHETIC_FRAME_FILENAME,
    TRUTH_FILENAME,
    load_simulation,
    simulate_replicate,
    simulation_dir,
)
from vocab_growth.workflow_cache import (
    checkpoint_matches,
    file_hashes,
    record_checkpoint,
    runtime_identity,
)


def simulation_inputs(
    model_key: str,
    definition,
    *,
    replicate: int,
    truth_source: str,
    truth_overrides,
    n_prior_draws: int,
    random_seed: int,
) -> dict:
    source = {}
    if truth_source == "posterior":
        directory = (
            Path(env.models_output_dir())
            / f"{definition.model_id}-{definition.config_name}"
        )
        require_current_fit(
            model_key,
            str(directory),
            consumer="fit_recovery.py resume",
            definition=definition,
        )
        source = file_hashes([directory / "fit_manifest.json"])
        trace = directory / "trace.nc"
        source["trace"] = {
            "size": trace.stat().st_size,
            "modified": trace.stat().st_mtime_ns,
        }
    return {
        "model": model_key,
        "definition": normalise_for_json(definition),
        "frame": expected_analysis_frame_hash(model_key, definition),
        "replicate": replicate,
        "truth_source": truth_source,
        "truth_overrides": [str(item) for item in truth_overrides],
        "n_prior_draws": n_prior_draws if truth_source == "prior" else None,
        "random_seed": random_seed,
        "source": source,
        "runtime": runtime_identity(),
    }


def recovery_fit_inputs(
    directory: Path, definition, fit_definition, config: str
) -> dict:
    return {
        "simulation": file_hashes(
            [
                directory / name
                for name in (
                    SIMULATION_FILENAME,
                    SYNTHETIC_FRAME_FILENAME,
                    TRUTH_FILENAME,
                )
            ]
        ),
        "generating_definition": normalise_for_json(definition),
        "fitted_definition": normalise_for_json(fit_definition),
        "config": config,
        "runtime": runtime_identity(),
    }


def recovery_fit_is_current(
    model_key: str,
    directory: Path,
    config: str,
    *,
    replicate: int,
    definition,
    fit_definition,
    truth_overrides,
    inputs: dict,
) -> bool:
    output = Path(
        recovery_fit_dir(
            model_key,
            replicate,
            definition=fit_definition,
            truth_definition=definition,
            truth_overrides=truth_overrides,
        )
    )
    if not checkpoint_matches(output / ".recovery-checkpoint.json", inputs):
        return False
    frame, truth, _ = load_simulation(str(directory), expected_definition=definition)
    truth.close()
    fitted = make_recovery_definition(
        fit_definition,
        replicate,
        truth_definition=definition,
        truth_overrides=truth_overrides,
    )
    policy = fit_validation_kwargs(
        "resume",
        expected_definition=fitted,
        expected_sampling_config_name=config,
        expected_sampling_parameters=asdict(
            sampling.get_sampling_configuration(config)
        ),
        current_git=git_metadata(env.ROOT_DIR),
        current_source_data_hash=source_data_hash(env.DATA_DIR),
        current_analysis_frame_hash=analysis_frame_hash(frame),
    )
    policy.update(require_clean_fit=True, require_convergence_evidence=True)
    return not validate_fit_output(str(output), **policy)


def run_recovery_stages(
    model_key: str,
    config: str,
    *,
    replicate: int,
    definition,
    fit_definition,
    truth_source: str,
    truth_overrides=(),
    n_prior_draws: int = 64,
    random_seed: int = 20260725,
    do_simulate: bool = True,
    do_fit: bool = True,
    fresh: bool = False,
) -> list[str]:
    """Run missing stages and report which complete stages were reused."""
    directory = Path(
        simulation_dir(definition, replicate, truth_overrides=truth_overrides)
    )
    reused = []
    if do_simulate:
        inputs = simulation_inputs(
            model_key,
            definition,
            replicate=replicate,
            truth_source=truth_source,
            truth_overrides=truth_overrides,
            n_prior_draws=n_prior_draws,
            random_seed=random_seed,
        )
        checkpoint = directory / ".simulation-checkpoint.json"
        if not fresh and checkpoint_matches(checkpoint, inputs):
            reused.append("simulation")
        else:
            checkpoint.unlink(missing_ok=True)
            simulate_replicate(
                model_key,
                config,
                replicate=replicate,
                truth_source=truth_source,
                truth_overrides=truth_overrides,
                n_prior_draws=n_prior_draws,
                random_seed=random_seed,
                definition=definition,
            )
            record_checkpoint(
                checkpoint,
                inputs,
                [
                    directory / name
                    for name in (
                        SIMULATION_FILENAME,
                        SYNTHETIC_FRAME_FILENAME,
                        TRUTH_FILENAME,
                    )
                ],
            )
    if do_fit:
        inputs = recovery_fit_inputs(directory, definition, fit_definition, config)
        current = not fresh and recovery_fit_is_current(
            model_key,
            directory,
            config,
            replicate=replicate,
            definition=definition,
            fit_definition=fit_definition,
            truth_overrides=truth_overrides,
            inputs=inputs,
        )
        if current:
            reused.append("fit")
        else:
            env.preflight_disk(
                20.0 if is_reporting_quality_config(config) else 2.0,
                env.output_root(),
                label=f"{model_key} recovery r{replicate:02d} [{config}]",
            )
            output = Path(
                recovery_fit_dir(
                    model_key,
                    replicate,
                    definition=fit_definition,
                    truth_definition=definition,
                    truth_overrides=truth_overrides,
                )
            )
            checkpoint = output / ".recovery-checkpoint.json"
            checkpoint.unlink(missing_ok=True)
            fit_recovery_replicate(
                model_key,
                config,
                replicate=replicate,
                definition=definition,
                fit_definition=fit_definition,
                truth_overrides=truth_overrides,
            )
            record_checkpoint(
                checkpoint,
                inputs,
                [
                    output / name
                    for name in (
                        "fit_manifest.json",
                        "fit_state.json",
                        "recovery_source.json",
                        "diagnostics_summary.json",
                    )
                ],
            )
    return reused
