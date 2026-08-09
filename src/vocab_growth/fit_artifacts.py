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
from enum import Enum, StrEnum
from pathlib import Path
from typing import Any, Literal

FIT_MANIFEST_FILENAME = "fit_manifest.json"
FIT_STATE_FILENAME = "fit_state.json"
TRACE_FILENAME = "trace.nc"
CONVERGENCE_FAILURE_FILENAME = "CONVERGENCE_FAILED.txt"
CONVERGENCE_CAVEATS_FILENAME = "CONVERGENCE_CAVEATS.txt"
"""Marker for a fit that cleared the hard gate but has sampling caveats.

Written by ``vocab_growth.models.common.enforce_convergence_gate`` when a
reporting-quality fit has divergent transitions or a low energy BFMI. Those are
soft-tier checks: they do not stop the fit being summarised and reported, but
they do stop it being published as a clean fit (see ``require_clean_convergence``
in :func:`validate_fit_output`).
"""

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
    "sync-with-caveats",
    "provisional-sync",
    "publish",
    "publish-with-caveats",
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

    Publication also requires clean convergence: a fit carrying soft-tier
    sampling caveats (divergences, low energy BFMI) stays usable for development
    and review, but must not be syndicated into the report as though it were
    clean. ``provisional-sync`` deliberately does not ask for this, so
    ``sync_report_figures.py --allow-provisional`` remains the way to work
    locally with a caveated fit.

    The ``-with-caveats`` purposes are the publication path for a fit that clears
    the hard tier but not the soft one. They keep **every** other publication
    check — reporting quality, rendered report, clean fit provenance, matching
    definition, sampling effort and raw-data fingerprint — and relax only the
    soft-tier requirement. They exist because the operative phrase above is *as
    though it were clean*: the objection is misrepresentation, not invalidity.
    ``methods-workflow.qmd`` §"Convergence diagnostics" states the same policy in
    the report's own words — such a fit "remains reportable ... but it is marked,
    and it cannot be syndicated into this report as a clean fit without that mark
    being carried with it" — and for several models in this family a handful of
    divergences or a BFMI slightly below 0.3 has not been removable without an
    infeasible reparameterisation. A blanket refusal is therefore *stricter than
    the documented policy*, and in practice blocks the whole report over the
    typically-developing hierarchical models' intrinsic BFMI.

    The mark is what makes this honest, so it is not optional: the caller must
    carry the caveats into the rendered output. ``sync_report_figures.py``
    discharges that by writing ``convergence_caveats.csv`` into the report figure
    cache, which Appendix B renders. Prefer plain ``publish``/``sync`` whenever a
    fit is clean; reach for these only for a fit whose caveats are being shown.
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
    elif purpose in {"sync", "publish", "sync-with-caveats", "publish-with-caveats"}:
        kwargs.update(
            require_reporting_quality=True,
            require_rendered_report=True,
            require_clean_fit=True,
            require_clean_convergence=not purpose.endswith("-with-caveats"),
        )
    elif purpose != "render":
        raise ValueError(f"Unknown fit-validation purpose: {purpose!r}.")
    return kwargs


DIAGNOSTICS_SUMMARY_FILENAME = "diagnostics_summary.json"


def convergence_caveats(gate_summary: dict | None) -> list[str]:
    """Soft-tier convergence problems recorded in a diagnostics-gate payload.

    ``write_diagnostics_summary`` (dse_research_utils) evaluates four checks. Two
    are **hard**: the R-hat/ESS scan is fail-closed in
    ``vocab_growth.models.common.enforce_convergence_gate``, because a fit that
    has not mixed cannot be summarised at all. The other two — divergent
    transitions and the energy BFMI — are **soft**: they indicate the sampler may
    have failed to traverse part of the posterior (so tail quantiles, i.e. the
    reported interval bounds, are the least trustworthy part of the fit), but a
    small number of divergences or a mildly low BFMI has been an accepted,
    recorded trade-off for this family rather than a bar to reporting (see
    ``notes/202607191614-full-refit-rep-hightune-run.md``).

    This lives here, beside the validators, because both ends need one
    implementation: the gate writes the caveats at fit time, and
    :func:`validate_fit_output` recomputes them from the payload on disk. Reading
    the payload rather than trusting a marker file means fits produced before the
    marker existed are assessed correctly too, instead of counting as clean
    because nothing ever looked.
    """
    if not gate_summary:
        return []
    checks = gate_summary.get("checks") or {}
    thresholds = gate_summary.get("thresholds") or {}
    caveats: list[str] = []

    if checks.get("divergences") is False:
        divergences = gate_summary.get("divergences")
        count = "an unknown number of" if divergences is None else f"{divergences}"
        caveats.append(
            f"{count} divergent transition(s): the sampler failed to traverse part "
            "of the posterior, so reported expectations may be biased."
        )
    if checks.get("bfmi") is False:
        per_chain = gate_summary.get("bfmi_per_chain") or []
        finite = [b for b in per_chain if b is not None]
        threshold = thresholds.get("bfmi_threshold")
        detail = f" (min {min(finite):.3f})" if finite else ""
        limit = "" if threshold is None else f" below {threshold}"
        caveats.append(
            f"energy BFMI{limit}{detail}: the energy chain explored the "
            "posterior's tails poorly, so the interval bounds are less reliable "
            "than the point estimates."
        )
    return caveats


def read_convergence_caveats(output_dir: str) -> list[str]:
    """Soft-tier caveats for a fit, read from its diagnostics payload.

    An unreadable or absent payload yields no caveats: the hard gate already
    fails closed when its own scan does not complete, so a missing summary is
    reported by the lifecycle checks rather than duplicated here.
    """
    path = os.path.join(output_dir, DIAGNOSTICS_SUMMARY_FILENAME)
    if not os.path.isfile(path):
        return []
    try:
        with open(path, encoding="utf-8") as handle:
            return convergence_caveats(json.load(handle))
    except (OSError, ValueError):
        return []


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
    require_clean_convergence: bool = False,
) -> list[str]:
    """Return every reason that fitted output is unsuitable for its intended use.

    ``require_clean_convergence`` additionally rejects a fit that cleared the hard
    convergence gate but recorded soft-tier caveats (divergent transitions or a low
    energy BFMI). Those caveats do not invalidate the fit for development or review
    — that is the project's recorded position — but publishing from one without
    saying so would misrepresent it, so the publication path asks for this while
    ``--allow-provisional`` does not.
    """
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
    if require_clean_convergence:
        # Read the diagnostics payload rather than the marker file, so a fit made
        # before the marker existed is judged on its actual diagnostics.
        for caveat in read_convergence_caveats(output_dir):
            errors.append(
                "Convergence caveat (cleared R-hat/ESS but not the soft tier): "
                + caveat.split(":", 1)[0]
                + "."
            )

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


# ----------------------------------------------------------------------------
# Trace persistence
# ----------------------------------------------------------------------------
# The posterior of these models is dominated by deterministic functions of the
# free parameters evaluated at every observation. On a reporting fit of VG03
# (10.94 GB, 4,075 observations, 6 x 6,000 draws) `f_obs`/`p_obs`/`kappa_obs`
# and the concatenated `f_all`/`g`/`g_unit` grids account for ~7 GB, while the
# free parameters and `sample_stats` together come to ~0.1 GB. Dropping the
# recomputable half is a storage policy and nothing more: it is applied when
# writing, to a copy, never to the in-memory trace that the rest of the fit
# pipeline goes on to use.
#
# See `notes/202608081445-trace-persistence-tiers.md`.


class TracePersistence(StrEnum):
    """How much of a fitted trace to persist to ``trace.nc``."""

    FULL = "full"
    """Store everything. The default, and what every fit before this did."""

    COMPACT = "compact"
    """Drop observation-sized posterior deterministics and the duplicated scaled
    random effects. ``log_likelihood`` and ``posterior_predictive`` are kept, so
    LOO and predictive checks can still be recomputed from the file."""

    MINIMAL = "minimal"
    """Additionally drop the observation-sized ``log_likelihood`` and
    ``posterior_predictive`` entries. Their consumers run during the fit and
    persist their own output, but recomputing LOO or a new predictive view later
    without refitting is no longer possible. Unlike ``COMPACT`` this is a real
    trade, not a free saving."""


# Resolved at call time with the same precedence as the output root: an explicit
# override (from `--trace-persistence`) > the environment variable > `full`.
#
# This deliberately does NOT live on `ModelDefinition`. Definition fields are
# part of the model graph and its fingerprint — `td_languages` is one precisely
# because changing it requires a refit — whereas how much of a trace is kept
# changes nothing about the posterior. Putting it there would invalidate every
# fitted model for a storage decision.
TRACE_PERSISTENCE_ENV_VAR = "DSE_VOCAB_GROWTH_TRACE_PERSISTENCE"

_trace_persistence_override: TracePersistence | None = None


def set_trace_persistence(value: TracePersistence | str | None) -> None:
    """Set a process-wide trace-persistence override (from ``--trace-persistence``).

    Takes precedence over ``$DSE_VOCAB_GROWTH_TRACE_PERSISTENCE``. Pass ``None``
    to clear it. Call once, early in a script's entry point, as with
    :func:`vocab_growth.environment.set_output_root`.
    """
    global _trace_persistence_override
    _trace_persistence_override = None if value is None else TracePersistence(value)


def configured_trace_persistence() -> TracePersistence:
    """Resolve the tier to use when a caller does not name one explicitly."""
    if _trace_persistence_override is not None:
        return _trace_persistence_override
    raw = os.environ.get(TRACE_PERSISTENCE_ENV_VAR)
    if not raw:
        return TracePersistence.FULL
    try:
        return TracePersistence(raw.strip().lower())
    except ValueError:
        valid = ", ".join(tier.value for tier in TracePersistence)
        raise ValueError(
            f"${TRACE_PERSISTENCE_ENV_VAR} is {raw!r}; expected one of {valid}."
        ) from None


# Filtering is scoped to named groups on purpose. `observed_data` (the counts)
# and `constant_data` (`X_obs`, and the `X_plot` grid the comparison suite
# reads) are both observation-dimensioned, so an unscoped dimension rule would
# delete the data itself; `sample_stats` is the sampler's own record and is
# what a convergence post-mortem needs.
_COMPACT_GROUPS = ("posterior",)
_MINIMAL_GROUPS = ("log_likelihood", "posterior_predictive")


def _is_observation_dim(dim: str) -> bool:
    """True for an observation-indexed dimension.

    Covers the joint models, whose ``log_likelihood`` is indexed per outcome
    (``obs_u_id``, ``obs_s_id``) even where their posterior uses a single
    ``obs_id``.
    """
    return dim == "obs_id" or (dim.startswith("obs_") and dim.endswith("_id"))


def _is_recomputable_dim(dim: str) -> bool:
    """True for dimensions whose variables are recomputable from the parameters.

    ``all_id`` is the concatenated obs+plot+query predictor grid (``f_all``,
    ``g``, ``g_unit``). It goes with the observation dimensions because the
    slices anything downstream actually reads are stored separately as
    ``*_plot`` and ``*_query``.
    """
    return dim == "all_id" or _is_observation_dim(dim)


def _group_dataset(trace: Any, group: str) -> Any | None:
    """Return ``group`` as an xarray Dataset, or ``None`` if absent."""
    try:
        node = trace[group]
    except (KeyError, TypeError):
        node = getattr(trace, group, None)
    if node is None:
        return None
    return node.to_dataset() if hasattr(node, "to_dataset") else node


def _droppable_variables(dataset: Any, *, drop_derived_effects: bool) -> list[str]:
    """Names in ``dataset`` that a non-``FULL`` tier would not persist."""
    present = set(dataset.data_vars)
    dropped: list[str] = []
    for name, variable in dataset.data_vars.items():
        if any(_is_recomputable_dim(dim) for dim in variable.dims):
            dropped.append(name)
        elif drop_derived_effects and f"{name}_raw" in present:
            # Non-centred random effects store both halves of `delta = tau *
            # delta_raw`. The raw draw and the scale are both kept, so the
            # scaled copy is exactly recoverable and need not be persisted.
            # Guarded on `_raw` being present: the centred branch samples
            # `delta` directly, and there the scaled copy is the only record.
            dropped.append(name)
    return sorted(dropped)


def plan_trace_persistence(
    trace: Any, persistence: TracePersistence | str = TracePersistence.FULL
) -> dict[str, list[str]]:
    """Return ``{group: [variable, ...]}`` that ``persistence`` would not persist.

    Pure: inspects the trace and decides, without writing anything. Separated
    from :func:`save_trace` so the policy can be tested, and reported in the fit
    manifest, without a file round-trip.
    """
    persistence = TracePersistence(persistence)
    if persistence is TracePersistence.FULL:
        return {}
    if _group_dataset(trace, "posterior") is None:
        # Without this, an object whose groups cannot be read yields an empty
        # plan and is then written in full — silently ignoring the requested
        # tier. A storage policy that quietly does nothing is worse than one
        # that fails, because the artifact looks correct.
        raise TypeError(
            f"Cannot apply persistence={persistence.value!r} to a trace of type "
            f"{type(trace).__name__}: it has no readable 'posterior' group."
        )
    groups = list(_COMPACT_GROUPS)
    if persistence is TracePersistence.MINIMAL:
        groups += list(_MINIMAL_GROUPS)
    plan: dict[str, list[str]] = {}
    for group in groups:
        dataset = _group_dataset(trace, group)
        if dataset is None:
            continue
        names = _droppable_variables(
            dataset, drop_derived_effects=group == "posterior"
        )
        if names:
            plan[group] = names
    return plan


def _filtered_trace(trace: Any, plan: dict[str, list[str]]) -> Any:
    """A copy of ``trace`` with ``plan``'s variables removed, leaving it unchanged."""
    # Imported here rather than at module scope: everything else in this module
    # is stdlib, and it is imported by tooling that has no other reason to pull
    # in xarray.
    import xarray as xr

    if not hasattr(trace, "children"):
        # Reached only after sampling has finished, so fail with something that
        # names the cause rather than an AttributeError on a several-hour fit.
        raise TypeError(
            f"Cannot filter a trace of type {type(trace).__name__}: expected an "
            "xarray DataTree (what ArviZ has used for every group since 1.0). "
            f"Save with persistence={TracePersistence.FULL.value!r} instead."
        )
    groups = {}
    for name, node in trace.children.items():
        dataset = node.to_dataset()
        drop = plan.get(name)
        groups[f"/{name}"] = dataset.drop_vars(drop) if drop else dataset
    filtered = xr.DataTree.from_dict(groups)
    filtered.attrs.update(trace.attrs)
    return filtered


def record_trace_persistence(output_dir: str, record: dict[str, Any]) -> bool:
    """Store what :func:`save_trace` actually wrote in the fit manifest.

    Recorded after the fact rather than when the manifest is first written,
    because that happens at the end of the fit's first stage — long before the
    trace exists, and before it is known whether a save was pinned to ``full``
    (the convergence-failure path is). A manifest stating the *intended* tier
    could therefore contradict the file beside it.

    Returns whether a manifest was found; a fit that writes none (VG17) is not
    an error.
    """
    path = os.path.join(output_dir, FIT_MANIFEST_FILENAME)
    if not os.path.isfile(path):
        return False
    try:
        manifest = read_json(path)
    except FitValidationError:
        return False
    manifest.setdefault("artefacts", {})["trace"] = record
    write_json_atomic(path, manifest)
    return True


def save_trace(
    trace: Any,
    output_dir: str,
    *,
    persistence: TracePersistence | str | None = None,
    filename: str = TRACE_FILENAME,
) -> dict[str, Any]:
    """Write ``trace`` to ``output_dir``, applying a persistence tier.

    The single place a fitted trace reaches disk, so the policy cannot drift
    between model engines. ``persistence`` defaults to
    :func:`configured_trace_persistence`, so the tier follows the process-wide
    setting without every engine having to thread it through; pass one
    explicitly to pin a particular save regardless of configuration.

    The record is written into ``fit_manifest.json`` as well as returned, so a
    later reader finding no ``f_obs`` can tell "dropped by policy" from
    "truncated or corrupt" — not a distinction to leave to guesswork, given a
    truncated trace is a failure this project has actually seen.

    ``trace`` is never modified — a non-``FULL`` tier filters a copy, because the
    fit pipeline goes on to read the in-memory trace after this returns.
    """
    persistence = (
        configured_trace_persistence()
        if persistence is None
        else TracePersistence(persistence)
    )
    plan = plan_trace_persistence(trace, persistence)
    path = os.path.join(output_dir, filename)
    (_filtered_trace(trace, plan) if plan else trace).to_netcdf(path)
    record = {
        "persistence": persistence.value,
        "dropped": plan,
        "dropped_count": sum(len(names) for names in plan.values()),
    }
    record_trace_persistence(output_dir, record)
    return record


def read_trace_persistence_record(output_dir: str) -> dict[str, Any] | None:
    """Return the ``artefacts.trace`` record from a fit's manifest, if it has one."""
    path = os.path.join(output_dir, FIT_MANIFEST_FILENAME)
    if not os.path.isfile(path):
        return None
    try:
        manifest = read_json(path)
    except FitValidationError:
        return None
    record = manifest.get("artefacts", {}).get("trace")
    return record if isinstance(record, dict) else None


def require_full_trace(output_dir: str, *, purpose: str) -> None:
    """Raise unless this fit's trace was persisted in full.

    For the consumers that need the observation-sized posterior or the stored
    log-likelihood — the cross-validation tools — which a ``compact`` or
    ``minimal`` fit does not carry. Checked from the manifest *before* the trace
    is opened, so a reporting-quality read of tens of gigabytes is not spent to
    arrive at a ``KeyError`` on a variable that was never going to be there.

    A fit with no persistence record predates the setting and was written in
    full, so it passes.
    """
    record = read_trace_persistence_record(output_dir)
    tier = (record or {}).get("persistence")
    if tier is None or tier == TracePersistence.FULL.value:
        return
    dropped = sorted(
        name
        for names in (record or {}).get("dropped", {}).values()
        for name in names
    )
    examples = ", ".join(dropped[:4]) + (" …" if len(dropped) > 4 else "")
    raise FitValidationError(
        f"{purpose} needs a trace saved in full, but {output_dir} was written "
        f"with trace persistence {tier!r}"
        + (f" (dropped {examples})" if dropped else "")
        + ". Refit with --trace-persistence full, or set "
        f"${TRACE_PERSISTENCE_ENV_VAR}=full."
    )
