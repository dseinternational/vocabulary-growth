# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Provenance manifests for cross-model comparison outputs (issue #266).

Comparison figures and tables are derived from per-model fitted output, but
until issue #266 they carried no record of *which* fits they were derived
from: ``sync_report_figures.py`` validated every model directory it copied and
then copied ``output/comparisons/`` wholesale, so a comparison generated from
a since-replaced fit was indistinguishable from a current one.

A comparison script records its provenance with :func:`write_comparison_manifest`
— one entry per script, merged into a single ``comparison_manifest.json`` in
the comparisons directory, naming its output files and fingerprinting every
contributing fit's ``fit_manifest.json``. The sync validates the manifest with
:func:`validate_comparison_manifest`: a fingerprint mismatch means a
contributing model was refitted after the comparison was generated, and the
comparison must be regenerated before it can be published.

Coverage is ratcheted rather than assumed: files in the comparisons directory
that no manifest entry claims are reported as warnings, so comparison scripts
that do not yet record provenance are visible without blocking the ones that
do. The nested ``recovery/`` and ``sensitivity/`` sub-directories are produced
by their own validated pipelines and are outside this manifest's scope.
"""

from __future__ import annotations

import hashlib
import os
from datetime import UTC, datetime

from vocab_growth.fit_artifacts import (
    FIT_MANIFEST_FILENAME,
    read_json,
    write_json_atomic,
)

COMPARISON_MANIFEST_FILENAME = "comparison_manifest.json"


def _file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def fit_manifest_fingerprint(model_output_dir: str) -> dict:
    """An identifying fingerprint of one contributing fit's manifest.

    The whole-file hash is the identity check — any refit rewrites the
    manifest (new ``created_at_utc`` at minimum). The frame and raw-data
    hashes are carried alongside so a mismatch report can say *what* moved.
    """
    manifest_path = os.path.join(model_output_dir, FIT_MANIFEST_FILENAME)
    manifest = read_json(manifest_path)
    data = manifest.get("data", {})
    return {
        "fit_manifest_sha256": _file_sha256(manifest_path),
        "created_at_utc": manifest.get("created_at_utc"),
        "analysis_frame_hash": data.get("analysis_frame_hash"),
        "source_data_hash": data.get("source_data_hash"),
    }


def write_comparison_manifest(
    comparisons_dir: str,
    *,
    script: str,
    contributing: dict[str, str],
    outputs: list[str],
) -> None:
    """Record one comparison script's provenance, merging with other scripts'.

    ``contributing`` maps a model label (``VG10-<config_name>``) to its fitted
    output directory; ``outputs`` names the files the script wrote into
    ``comparisons_dir`` (basenames).
    """
    manifest_path = os.path.join(comparisons_dir, COMPARISON_MANIFEST_FILENAME)
    payload: dict = {"schema_version": 1, "scripts": {}}
    if os.path.isfile(manifest_path):
        payload = read_json(manifest_path)
        payload.setdefault("scripts", {})
    payload["scripts"][script] = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "outputs": sorted(outputs),
        "contributing_fits": {
            label: fit_manifest_fingerprint(model_dir)
            for label, model_dir in sorted(contributing.items())
        },
    }
    write_json_atomic(manifest_path, payload)


def validate_comparison_manifest(
    comparisons_dir: str, models_dir: str
) -> tuple[list[str], list[str]]:
    """Validate every recorded comparison against the current fitted output.

    Returns ``(errors, warnings)``. Errors: the manifest is missing or
    unreadable, a contributing fit's directory or manifest is gone, or a
    contributing fit's manifest no longer matches its recorded fingerprint
    (the model was refitted after the comparison was generated). Warnings:
    files in the comparisons directory that no manifest entry claims.
    """
    errors: list[str] = []
    warnings: list[str] = []
    manifest_path = os.path.join(comparisons_dir, COMPARISON_MANIFEST_FILENAME)
    if not os.path.isfile(manifest_path):
        return (
            [
                f"{COMPARISON_MANIFEST_FILENAME} is missing from "
                f"{comparisons_dir}: regenerate the comparisons with the "
                "current scripts so their contributing fits are recorded."
            ],
            warnings,
        )
    try:
        payload = read_json(manifest_path)
    except Exception as exc:  # noqa: BLE001 - reported, not raised
        return [f"Could not read {manifest_path}: {exc}"], warnings

    claimed: set[str] = set()
    for script, entry in sorted((payload.get("scripts") or {}).items()):
        claimed.update(entry.get("outputs") or [])
        for label, recorded in sorted(
            (entry.get("contributing_fits") or {}).items()
        ):
            model_dir = os.path.join(models_dir, label)
            manifest_file = os.path.join(model_dir, FIT_MANIFEST_FILENAME)
            if not os.path.isfile(manifest_file):
                errors.append(
                    f"{script}: contributing fit {label} has no "
                    f"{FIT_MANIFEST_FILENAME} under {models_dir}."
                )
                continue
            if _file_sha256(manifest_file) != recorded.get("fit_manifest_sha256"):
                errors.append(
                    f"{script}: contributing fit {label} was refitted after "
                    "this comparison was generated; regenerate the comparison."
                )

    for name in sorted(os.listdir(comparisons_dir)):
        path = os.path.join(comparisons_dir, name)
        if not os.path.isfile(path) or name == COMPARISON_MANIFEST_FILENAME:
            continue
        if name not in claimed:
            warnings.append(
                f"comparison file {name} is not claimed by any "
                f"{COMPARISON_MANIFEST_FILENAME} entry; its provenance is "
                "unrecorded."
            )
    return errors, warnings
