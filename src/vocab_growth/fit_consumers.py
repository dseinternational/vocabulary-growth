# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Check stored fits against the current definition and prepared data.

Consumers compare the registered definition, raw-data fingerprint and exact
prepared-frame hash before reading a trace. A frame hash detects loader-rule
changes that the raw-data fingerprint cannot detect.

These checks do not require publication-quality sampling, a rendered report,
a clean checkout or a matching executable signature. They therefore establish
definition and data compatibility, not that the current likelihood code is
unchanged or the fit is publishable. Publication uses stricter validation.

``--allow-stale-fit`` prints any bypassed errors. ``EXEMPT_CONSUMERS`` documents
scripts that use their own validation path or inspect files without reading
posterior values.
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

#: Scripts with a separate validation path or no posterior-value reads.
EXEMPT_CONSUMERS: dict[str, str] = {
    "fit_recovery.py": (
        "validates, but not here and not against the registry. The trace it "
        "opens itself is a recovery replicate's own fit, in a variant directory "
        "whose definition is meant to differ. The model of record it takes a "
        "truth from is opened inside `recovery.simulate.truth_from_trace`, "
        "which makes these same three checks and refuses before slicing a "
        "multi-gigabyte trace, with the remedy this module cannot offer: refit "
        "the model of record, or use `--truth prior`. `load_simulation` "
        "separately compares the definition a simulation recorded against the "
        "one about to fit it, which is what the staged `--fit-only` path needs."
    ),
    "compact_traces.py": (
        "manipulates trace files as files -- it drops recomputable variables and "
        "never reads a posterior value into a reported number. It carries its own "
        "mid-promotion liveness guard instead, which is the risk actually present "
        "when rewriting a fit in place."
    ),
    "resume_comparison.py": (
        "checks trace file metadata without opening posterior values. Its child "
        "comparison script validates fits before generating any result. Reuse "
        "requires the same recorded comparison entry, code, data and hashed "
        "outputs, and the same manifests, lifecycle files, summary tables and "
        "trace metadata for every fit in the output root. "
        "The replication driver and report sync retain their validation gates."
    ),
}

_frame_hashes: dict[str, str] = {}
_source_hashes: dict[str, str] = {}


def _current_frame_hash(model_key: str, definition: ModelDefinition) -> str:
    """``expected_analysis_frame_hash``, memoised per model key.

    Repeated checks of one model reuse its frame hash instead of reloading data.
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
    """Validation errors for ``fit_dir`` against ``model_key``'s definition and data.

    An empty list means the required artefacts, definition and prepared data
    pass. This check does not compare executable signatures.
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

    Returns the ``{label: output_dir}`` mapping recorded by
    :func:`vocab_growth.comparisons_provenance.write_comparison_manifest`.
    The same call validates fits and identifies them for provenance.
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
