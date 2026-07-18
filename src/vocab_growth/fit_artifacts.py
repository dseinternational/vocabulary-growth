# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Lifecycle, provenance, and publication checks for fitted model artefacts.

A ``trace.nc`` file alone does not show that a fit completed successfully. It
can exist after interrupted sampling, failed convergence checks, or a fast
development run. This module gives every fit an explicit state and provides one
validator for resuming, synchronising, and publishing model output.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import uuid
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any, Literal

FIT_MANIFEST_FILENAME = "fit_manifest.json"
FIT_STATE_FILENAME = "fit_state.json"
CONVERGENCE_FAILURE_FILENAME = "CONVERGENCE_FAILED.txt"

REPORTING_CONFIGS = {
    "reporting",
    "report",
    "rep",
    "rep-lite",
    "reporting-lite",
    "rep_lite",
}
NON_REPORTING_CONFIGS = {"dev", "development", "test", "testing"}


class FitValidationError(RuntimeError):
    """Raised when fitted output is incomplete or incompatible with its use."""


def _json_default(value: Any) -> Any:
    """Convert model configuration values to stable JSON representations."""
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return str(value)


def normalise_for_json(value: Any) -> Any:
    """Return ``value`` as the same plain data structures used in manifests."""
    if is_dataclass(value) and not isinstance(value, type):
        value = asdict(value)
    return json.loads(json.dumps(value, sort_keys=True, default=_json_default))


def read_json(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Read a JSON object, giving a useful error for malformed metadata."""
    try:
        with open(path, encoding="utf-8") as source:
            payload = json.load(source)
    except (OSError, json.JSONDecodeError) as exc:
        raise FitValidationError(f"Could not read {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise FitValidationError(f"Expected a JSON object in {path}.")
    return payload


def write_json_atomic(path: str, payload: dict[str, Any]) -> None:
    """Write metadata atomically so interruption cannot leave partial JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = f"{path}.{uuid.uuid4().hex}.tmp"
    with open(temporary, "w", encoding="utf-8") as destination:
        json.dump(payload, destination, indent=2, sort_keys=True, default=_json_default)
        destination.write("\n")
    os.replace(temporary, path)


def write_fit_state(
    output_dir: str,
    state: str,
    *,
    model_id: str,
    config_name: str,
    sampling_config_name: str,
    stage: str | None = None,
    error: BaseException | None = None,
) -> None:
    """Record the current lifecycle state of one fit.

    ``complete`` is the only state accepted by downstream publication tools.
    Failed staging output is retained separately for diagnosis.
    """
    valid_states = {"initialising", "running", "complete", "failed"}
    if state not in valid_states:
        raise ValueError(f"Unknown fit state {state!r}; expected one of {valid_states}.")

    path = os.path.join(output_dir, FIT_STATE_FILENAME)
    now = datetime.now(UTC).isoformat()
    started_at = now
    previous_stage = None
    if os.path.isfile(path):
        try:
            previous = read_json(path)
            started_at = previous.get("started_at_utc", now)
            previous_stage = previous.get("stage")
        except FitValidationError:
            pass

    payload: dict[str, Any] = {
        "schema_version": 1,
        "state": state,
        "stage": previous_stage if stage is None else stage,
        "model_id": model_id,
        "config_name": config_name,
        "sampling_configuration": sampling_config_name,
        "started_at_utc": started_at,
        "updated_at_utc": now,
        "completed_at_utc": now if state == "complete" else None,
    }
    if error is not None:
        payload["error"] = {
            "type": type(error).__name__,
            "message": str(error),
        }
    write_json_atomic(path, payload)


def git_metadata(repo_dir: str) -> dict[str, object]:
    """Return the repository revision and dirty state without requiring Git."""

    def run_git(*args: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=repo_dir,
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return result.stdout.strip()

    commit = run_git("rev-parse", "HEAD")
    branch_result = run_git("branch", "--show-current")
    if branch_result is None:
        branch = None
        detached = None
    else:
        branch = branch_result or None
        detached = not bool(branch_result)
    status = run_git("status", "--porcelain", "--untracked-files=normal")
    return {
        "commit": commit,
        "branch": branch,
        "detached": detached,
        "dirty": None if status is None else bool(status),
    }


def source_data_hash(data_dir: str) -> str:
    """Hash the raw CSV inputs that determine the prepared analysis data."""
    digest = hashlib.sha256()
    paths = sorted(
        path
        for path in Path(data_dir).glob("*.csv")
        if path.name != "vocab_data_merged.csv"
    )
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return f"sha256:{digest.hexdigest()}"


def is_reporting_quality_config(sampling_config_name: str) -> bool:
    """Classify a known sampling tier without silently accepting new aliases."""
    name = sampling_config_name.strip().lower()
    if name in REPORTING_CONFIGS:
        return True
    if name in NON_REPORTING_CONFIGS:
        return False
    raise ValueError(
        f"Sampling configuration {sampling_config_name!r} has no convergence-gate "
        "classification. Add it explicitly before fitting or publication."
    )


def _sampling_parameter_errors(
    recorded: Any,
    expected: Any,
) -> list[str]:
    """Compare sampling settings by statistical strength, not machine layout.

    ``cores`` controls local parallelism but does not change the requested
    chains or draws, so it is deliberately ignored. Reporting fits may use
    documented high-tuning overrides; draws, tuning iterations, chains and
    target acceptance are therefore minimum requirements. Other settings,
    including the random seed, must match exactly.
    """
    recorded_payload = normalise_for_json(recorded)
    expected_payload = normalise_for_json(expected)
    if not isinstance(recorded_payload, dict) or not isinstance(expected_payload, dict):
        return ["Sampling parameters are not recorded as a JSON object."]

    errors: list[str] = []
    minimum_fields = {"draws", "tune", "chains", "target_accept"}
    ignored_fields = {"cores"}
    for field in sorted(minimum_fields):
        recorded_value = recorded_payload.get(field)
        expected_value = expected_payload.get(field)
        if not isinstance(recorded_value, (int, float)) or not isinstance(
            expected_value, (int, float)
        ):
            errors.append(f"Sampling parameter {field!r} is missing or non-numeric.")
        elif recorded_value < expected_value:
            errors.append(
                f"Sampling parameter {field!r} is {recorded_value!r}; "
                f"at least {expected_value!r} is required."
            )

    exact_fields = set(expected_payload) - minimum_fields - ignored_fields
    for field in sorted(exact_fields):
        if recorded_payload.get(field) != expected_payload.get(field):
            errors.append(
                f"Sampling parameter {field!r} is {recorded_payload.get(field)!r}; "
                f"expected {expected_payload.get(field)!r}."
            )
    return errors


FitValidationPurpose = Literal[
    "resume",
    "render",
    "sync",
    "provisional-sync",
    "publish",
]


def fit_validation_kwargs(
    purpose: FitValidationPurpose,
    *,
    expected_definition: Any,
    expected_sampling_config_name: str,
    expected_sampling_parameters: Any,
    current_git: dict[str, object] | None = None,
    current_source_data_hash: str | None = None,
) -> dict[str, Any]:
    """Build one documented validation policy for each artefact consumer.

    Resume is intentionally strict about the current code and raw data because
    downstream computations would otherwise mix revisions. Publication instead
    checks that the fit itself came from a clean revision; later documentation
    commits do not invalidate an already complete fit. Provisional local syncs
    retain lifecycle/model/sampling checks while allowing exploratory code and
    data changes.
    """
    kwargs: dict[str, Any] = {
        "expected_definition": expected_definition,
        "expected_sampling_config_name": expected_sampling_config_name,
        "expected_sampling_parameters": expected_sampling_parameters,
    }
    if purpose == "provisional-sync":
        return kwargs

    if current_source_data_hash is None:
        raise ValueError(f"{purpose} validation requires the current source-data hash.")
    kwargs["expected_source_data_hash"] = current_source_data_hash

    if purpose == "resume":
        if current_git is None:
            raise ValueError("resume validation requires current Git metadata.")
        kwargs.update(
            expected_git=current_git,
            require_clean_checkout=True,
        )
    elif purpose in {"sync", "publish"}:
        kwargs.update(
            require_reporting_quality=True,
            require_rendered_report=True,
            require_clean_fit=True,
        )
    elif purpose != "render":
        raise ValueError(f"Unknown fit-validation purpose: {purpose!r}.")
    return kwargs


def validate_fit_output(
    output_dir: str,
    *,
    expected_definition: Any | None = None,
    expected_sampling_config_name: str | None = None,
    expected_sampling_parameters: Any | None = None,
    expected_git: dict[str, object] | None = None,
    expected_source_data_hash: str | None = None,
    require_reporting_quality: bool = False,
    require_rendered_report: bool = False,
    require_clean_fit: bool = False,
    require_clean_checkout: bool = False,
) -> list[str]:
    """Return every reason that fitted output is unsuitable for its intended use."""
    errors: list[str] = []
    state_path = os.path.join(output_dir, FIT_STATE_FILENAME)
    manifest_path = os.path.join(output_dir, FIT_MANIFEST_FILENAME)

    try:
        state = read_json(state_path)
    except FitValidationError as exc:
        state = {}
        errors.append(str(exc))
    if state and state.get("state") != "complete":
        errors.append(f"Fit state is {state.get('state')!r}, not 'complete'.")

    try:
        manifest = read_json(manifest_path)
    except FitValidationError as exc:
        manifest = {}
        errors.append(str(exc))

    trace_path = os.path.join(output_dir, "trace.nc")
    if not os.path.isfile(trace_path):
        errors.append("trace.nc is missing.")
    if os.path.isfile(os.path.join(output_dir, CONVERGENCE_FAILURE_FILENAME)):
        errors.append(f"{CONVERGENCE_FAILURE_FILENAME} is present.")

    if not manifest:
        return errors

    manifest_definition = manifest.get("model", {}).get("definition")
    if expected_definition is not None and manifest_definition != normalise_for_json(
        expected_definition
    ):
        errors.append("The model definition differs from the current registered definition.")

    sampling_payload = manifest.get("sampling", {})
    recorded_sampling_name = sampling_payload.get("configuration_name")
    if (
        expected_sampling_config_name is not None
        and recorded_sampling_name != expected_sampling_config_name
    ):
        errors.append(
            "Sampling configuration mismatch: "
            f"found {recorded_sampling_name!r}, expected {expected_sampling_config_name!r}."
        )
    if expected_sampling_parameters is not None:
        errors.extend(
            _sampling_parameter_errors(
                sampling_payload.get("parameters"), expected_sampling_parameters
            )
        )

    data_payload = manifest.get("data", {})
    if (
        expected_source_data_hash is not None
        and data_payload.get("source_data_hash") != expected_source_data_hash
    ):
        errors.append("Raw data inputs differ from those used for this fit.")

    code_payload = manifest.get("code", {})
    if expected_git is not None:
        if code_payload.get("commit") != expected_git.get("commit"):
            errors.append("The current Git commit differs from the commit used for this fit.")
    if require_clean_checkout:
        if expected_git is None or expected_git.get("dirty") is not False:
            errors.append("The current checkout is dirty, so an exact resume is unsafe.")
    if require_clean_fit and code_payload.get("dirty") is not False:
        errors.append("The fit was produced from a dirty or unverifiable checkout.")

    if require_reporting_quality:
        try:
            reporting_quality = is_reporting_quality_config(str(recorded_sampling_name))
        except ValueError as exc:
            errors.append(str(exc))
        else:
            if not reporting_quality:
                errors.append(
                    f"Sampling configuration {recorded_sampling_name!r} is not reporting-quality."
                )

    if require_rendered_report and not os.path.isfile(
        os.path.join(output_dir, "index.html")
    ):
        errors.append("Rendered report index.html is missing.")

    return errors


def require_valid_fit(output_dir: str, **kwargs: Any) -> None:
    """Raise :class:`FitValidationError` if :func:`validate_fit_output` fails."""
    errors = validate_fit_output(output_dir, **kwargs)
    if errors:
        detail = "\n - ".join(errors)
        raise FitValidationError(f"Fit output is not valid for this operation:\n - {detail}")


def create_staging_root(output_root: str, model_label: str) -> str:
    """Create an isolated root for a fit before atomic publication."""
    safe_label = model_label.replace(os.sep, "-")
    run_id = f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    staging_root = os.path.join(output_root, ".staging", f"{safe_label}-{run_id}")
    os.makedirs(staging_root, exist_ok=False)
    return staging_root


def promote_staged_fit(staged_output_dir: str, canonical_output_dir: str) -> None:
    """Replace canonical output only after a staged fit has fully completed."""
    parent = os.path.dirname(canonical_output_dir)
    output_root = os.path.dirname(parent)
    os.makedirs(parent, exist_ok=True)
    previous_root = os.path.join(output_root, ".previous")
    os.makedirs(previous_root, exist_ok=True)
    backup = os.path.join(
        previous_root,
        f"{os.path.basename(canonical_output_dir)}-{uuid.uuid4().hex[:8]}",
    )

    had_canonical = os.path.exists(canonical_output_dir)
    if had_canonical:
        os.replace(canonical_output_dir, backup)
    try:
        os.replace(staged_output_dir, canonical_output_dir)
    except BaseException:
        if had_canonical and os.path.exists(backup):
            os.replace(backup, canonical_output_dir)
        raise
    if had_canonical and os.path.exists(backup):
        shutil.rmtree(backup)


def retain_failed_fit(staged_output_dir: str, output_root: str) -> str | None:
    """Move failed staged output aside for diagnosis and return its new path."""
    if not os.path.isdir(staged_output_dir):
        return None
    failed_root = os.path.join(output_root, "failed")
    os.makedirs(failed_root, exist_ok=True)
    destination = os.path.join(
        failed_root,
        f"{os.path.basename(staged_output_dir)}-{datetime.now(UTC):%Y%m%dT%H%M%SZ}",
    )
    if os.path.exists(destination):
        destination = f"{destination}-{uuid.uuid4().hex[:8]}"
    os.replace(staged_output_dir, destination)
    return destination
