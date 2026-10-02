# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Content checks for repeatable workflow steps, separate from fit validation."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from importlib import metadata
from pathlib import Path

from vocab_growth.fit_artifacts import write_json_atomic
from vocab_growth.models.implementation_identity import implementation_signature


def file_hash(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_hashes(paths) -> dict[str, str]:
    return {
        str(Path(path).resolve()): file_hash(path)
        for path in sorted(set(map(Path, paths)))
    }


def runtime_identity() -> dict:
    # These additional libraries affect compilation, simulation and NetCDF I/O.
    # The established implementation signature supplies the model code and the
    # main numerical libraries, including Git origins for the shared library.
    versions = {}
    for name in ("numba", "llvmlite", "jax", "jaxlib", "xarray", "h5netcdf", "h5py"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return {
        "implementation": implementation_signature(),
        "python": sys.version,
        "libraries": versions,
        "settings": {
            name: os.environ.get(name)
            for name in (
                "PYTENSOR_FLAGS",
                "NUMBA_CPU_NAME",
                "NUMBA_CPU_FEATURES",
                "DSE_VOCAB_GROWTH_NUTPIE_BACKEND",
                "DSE_VOCAB_GROWTH_TRACE_PERSISTENCE",
            )
        },
    }


def checkpoint_matches(path: str | Path, inputs: dict) -> bool:
    """A checkpoint is usable only when its inputs and every output still match."""
    try:
        record = json.loads(Path(path).read_text(encoding="utf-8"))
        outputs = record["outputs"]
        return (
            record.get("schema_version") == 1
            and record.get("inputs") == inputs
            and isinstance(outputs, dict)
            and bool(outputs)
            and all(file_hash(name) == value for name, value in outputs.items())
        )
    except OSError, ValueError, KeyError, TypeError, AttributeError:
        return False


def record_checkpoint(path: str | Path, inputs: dict, outputs) -> None:
    """Record only a successful step with present, hashed output files."""
    hashes = file_hashes(outputs)
    if not hashes:
        raise ValueError("A workflow checkpoint needs at least one output.")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(
        str(path), {"schema_version": 1, "inputs": inputs, "outputs": hashes}
    )
