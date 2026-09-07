# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""One frame-checked way to read a fit back (issue #266 finding 1).

The manifest records ``data.analysis_frame_hash``, an exact hash of the prepared
frame, and :func:`vocab_growth.analysis_frames.expected_analysis_frame_hash`
recomputes what that hash would be today. Finding 1 asked for the comparison to
run "for every fit consumer"; the first pass wired it into the fit pipeline and
the publication path, which left the scripts that open a stored trace and print a
number from it unchecked. Eleven of the sixteen were, and nine of those produce a
reported quantity from a posterior never compared against the current loader
rules -- rules that run in Python *after* the CSVs are read, so the raw-data
fingerprint some of them did carry cannot see them move.

The check here is deliberately the same one ``scripts/loso_compare.py`` already
made, rather than a new policy: the registered definition, the raw-data
fingerprint and the exact frame hash. Those three are the identity of the graph
and of the data it was fitted to. What a consumer does *not* ask for is the
publication apparatus -- reporting-quality sampling, a rendered report, a clean
checkout, the executable-code signature. A script that prints a number off a
model of record is not publishing it, and requiring the code signature would mean
an edit to any module in the package stopped every one of these scripts from
running, which is the reasoning ``fit_validation_kwargs`` already records for
``render``.

Each consumer names itself, so a refusal says which script refused and what to do
about it, and each offers ``--allow-stale-fit`` to override -- following
``sync_report_figures.py --allow-provisional``, the established way to work
locally with output that will not pass publication. An override prints what it
overrode, every time: reading a superseded posterior should be a choice someone
made rather than something that happened quietly.

:data:`EXEMPT_CONSUMERS` records the two scripts that read a trace and are *not*
expected to validate, with the reason. An exemption that is written down can be
argued with; an absence looks like an oversight, which is what the other eleven
turned out to be.
"""

from __future__ import annotations

import argparse
import os
from typing import Any

from vocab_growth import environment as env
from vocab_growth.analysis_frames import expected_analysis_frame_hash
from vocab_growth.fit_artifacts import (
    FitValidationError,
    source_data_hash,
    validate_fit_output,
)
from vocab_growth.models.definitions import MODEL_REGISTRY, ModelDefinition

#: Trace-reading scripts that deliberately do not validate, and why. Both were
#: argued for on issue #266 on 2026-09-06 and are recorded here rather than left
#: as an absence, the way ``fit_validation_kwargs`` records which purposes omit
#: the executable-code signature.
EXEMPT_CONSUMERS: dict[str, str] = {
    "fit_recovery.py": (
        "reads a posterior only as a truth generator: a draw from a fit whose "
        "frame has since moved is still a valid parameter vector to simulate "
        "from, and the refit it is scored against is made by the current "
        "pipeline under the current rules. Staleness does not make the truth "
        "worse."
    ),
    "compact_traces.py": (
        "manipulates trace files as files -- it drops recomputable variables and "
        "never reads a posterior value into a reported number. It carries its own "
        "mid-promotion liveness guard instead, which is the risk actually present "
        "when rewriting a fit in place."
    ),
}

_frame_hashes: dict[str, str] = {}
_source_hashes: dict[str, str] = {}


def _current_frame_hash(model_key: str, definition: ModelDefinition) -> str:
    """``expected_analysis_frame_hash``, memoised per model key.

    Rebuilding a frame reloads and re-prepares the pool, so a script checking
    eighteen models would otherwise pay for eighteen rebuilds.
    """
    key = model_key.lower()
    if key not in _frame_hashes:
        _frame_hashes[key] = expected_analysis_frame_hash(key, definition)
    return _frame_hashes[key]


def _current_source_hash() -> str:
    key = env.DATA_DIR
    if key not in _source_hashes:
        _source_hashes[key] = source_data_hash(key)
    return _source_hashes[key]


def model_fit_dir(model_key: str, *, suffix: str | None = None) -> str:
    """The canonical output directory for a registered model's fit."""
    definition = MODEL_REGISTRY[model_key.lower()]
    name = f"{definition.model_id}-{definition.config_name}"
    if suffix:
        name = f"{name}-{suffix}"
    return os.path.join(env.models_output_dir(), name)


def model_key_for_dir(fit_dir: str) -> str | None:
    """The registered model whose canonical output directory this is, if any.

    ``None`` for a sensitivity, recovery or fold directory. Those carry a
    definition that is *supposed* to differ from the registered one, so
    validating them against the registry would report a difference the variant
    exists to make. Their provenance is the job of the pipeline that produced
    them: ``compare_sensitivity.py`` pairs each variant against the baseline it
    was fitted from, and the recovery harness scores only replicates whose own
    fit converged.
    """
    name = os.path.basename(os.path.normpath(fit_dir)).lower()
    for key, definition in MODEL_REGISTRY.items():
        if f"{definition.model_id}-{definition.config_name}".lower() == name:
            return key
    return None


def fit_errors(
    model_key: str,
    fit_dir: str | None = None,
    *,
    definition: ModelDefinition | None = None,
) -> list[str]:
    """Every reason ``fit_dir`` is not a current fit of ``model_key``.

    Empty means the stored posterior was fitted from the registered definition
    on the frame today's loader rules produce.
    """
    key = model_key.lower()
    if definition is None:
        definition = MODEL_REGISTRY[key]
    directory = model_fit_dir(key) if fit_dir is None else fit_dir
    if not os.path.isdir(directory):
        return [f"No fitted output at {directory}."]
    return validate_fit_output(
        directory,
        expected_definition=definition,
        expected_source_data_hash=_current_source_hash(),
        expected_analysis_frame_hash=_current_frame_hash(key, definition),
    )


def require_current_fit(
    model_key: str,
    fit_dir: str | None = None,
    *,
    consumer: str,
    allow_stale: bool = False,
    definition: ModelDefinition | None = None,
) -> list[str]:
    """Refuse to read a fit that is not current, unless told to anyway.

    Returns the errors found, so a caller that passed ``allow_stale`` can carry
    them into whatever it writes. Raises :class:`FitValidationError` otherwise.
    """
    errors = fit_errors(model_key, fit_dir, definition=definition)
    if not errors:
        return []
    directory = model_fit_dir(model_key) if fit_dir is None else fit_dir
    detail = "\n  - ".join(errors)
    if allow_stale:
        print(
            f"[stale fit accepted] {consumer}: {model_key} at {directory}\n"
            f"  - {detail}\n"
            "  Numbers produced from this fit do not describe the current data "
            "or definition.",
            flush=True,
        )
        return errors
    raise FitValidationError(
        f"{consumer} will not read {model_key} at {directory}: it is not a "
        f"current fit.\n  - {detail}\n"
        "  Refit the model, or pass --allow-stale-fit to read it anyway and "
        "accept that the numbers describe a superseded fit."
    )


def require_current_fit_dir(
    fit_dir: str, *, consumer: str, allow_stale: bool = False
) -> list[str]:
    """:func:`require_current_fit` for a caller that has a directory, not a key.

    A directory that is not a registered model's canonical output says so on
    stdout rather than passing silently, so "not checked" and "checked and
    clean" never look the same in a log.
    """
    key = model_key_for_dir(fit_dir)
    if key is None:
        print(
            f"[provenance not checked] {consumer}: {fit_dir} is not a registered "
            "model of record; its definition is expected to differ from the "
            "registry, so it is validated by the pipeline that produced it.",
            flush=True,
        )
        return []
    return require_current_fit(
        key, fit_dir, consumer=consumer, allow_stale=allow_stale
    )


def load_validated_trace(
    model_key: str,
    fit_dir: str | None = None,
    *,
    consumer: str,
    allow_stale: bool = False,
    definition: ModelDefinition | None = None,
) -> Any:
    """The trace at ``fit_dir``, after :func:`require_current_fit` accepts it."""
    import arviz as az

    require_current_fit(
        model_key,
        fit_dir,
        consumer=consumer,
        allow_stale=allow_stale,
        definition=definition,
    )
    directory = model_fit_dir(model_key) if fit_dir is None else fit_dir
    return az.from_netcdf(os.path.join(directory, "trace.nc"))


def contributing_fits(
    model_keys: list[str] | tuple[str, ...],
    *,
    consumer: str,
    allow_stale: bool = False,
) -> dict[str, str]:
    """Validate every fit a comparison reads, and name them for its manifest.

    Returns the ``{label: output_dir}`` mapping
    :func:`vocab_growth.comparisons_provenance.write_comparison_manifest`
    records, so the two halves of issue #266 -- "validate what you read" and
    "record what you read" -- are discharged by one call and cannot drift apart.
    A comparison whose manifest lists a fit it did not check, or checks one it
    does not list, is the exact failure the manifest exists to prevent.
    """
    contributing: dict[str, str] = {}
    # A script may name the same model twice for different roles --
    # ``compare_ds_td_expressive.py`` reads VG20 as both its joint and its
    # dispersion comparator -- so validate each fit once.
    for key in dict.fromkeys(k.lower() for k in model_keys):
        directory = model_fit_dir(key)
        require_current_fit(
            key, directory, consumer=consumer, allow_stale=allow_stale
        )
        contributing[os.path.basename(directory)] = directory
    return contributing


def add_allow_stale_argument(parser: argparse.ArgumentParser) -> None:
    """Add the shared ``--allow-stale-fit`` override to a consumer's parser."""
    parser.add_argument(
        "--allow-stale-fit",
        action="store_true",
        help=(
            "Read a fit whose definition, raw data or prepared frame no longer "
            "matches the current checkout. The mismatch is printed; the numbers "
            "produced then describe the superseded fit."
        ),
    )
