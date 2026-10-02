# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Run a comparison unless a checked checkpoint covers this exact invocation."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from vocab_growth import environment as env
from vocab_growth.comparisons_provenance import (
    COMPARISON_MANIFEST_FILENAME,
    comparison_code_signature,
)
from vocab_growth.fit_artifacts import source_data_hash
from vocab_growth.workflow_cache import (
    checkpoint_matches,
    file_hashes,
    record_checkpoint,
    runtime_identity,
)


def _fit_inputs(fit: Path) -> dict:
    trace = fit / "trace.nc"
    return {
        "files": file_hashes(
            [
                p
                for p in fit.iterdir()
                if p.is_file()
                and p.suffix in {".csv", ".json"}
                and p.name != ".render-checkpoint.json"
            ]
        ),
        "trace": {
            "size": trace.stat().st_size,
            "modified": trace.stat().st_mtime_ns,
        }
        if trace.exists()
        else None,
    }


def comparison_inputs(script: str, arguments: list[str]) -> tuple[dict, list[Path]]:
    directory = Path(env.comparisons_output_dir())
    record = json.loads((directory / COMPARISON_MANIFEST_FILENAME).read_text())
    entry = record["scripts"][script]
    if not entry.get("outputs") or not any(
        entry.get(key)
        for key in ("contributing_fits", "source_files", "source_data_hash")
    ):
        raise ValueError("Comparison has no recorded inputs or outputs")
    models = Path(env.models_output_dir())
    files = []
    for label in entry.get("contributing_fits", {}):
        if Path(label).name != label:
            raise ValueError("Invalid contributing fit path")
        files.append(models / label / "fit_manifest.json")
    # Every fit on disk, not only the recorded contributors: loo_compare.py and
    # prior_vs_posterior.py skip a model without a usable trace, so a fit that
    # appears or changes after the checkpoint must rerun the comparison.
    fits = {
        fit.name: _fit_inputs(fit)
        for fit in (sorted(models.iterdir()) if models.is_dir() else [])
        if fit.is_dir() and not fit.name.startswith(".")
    }
    for item in entry.get("source_files", {}).values():
        files.append(Path(env.ROOT_DIR) / item["path"])
    outputs = []
    for name in entry["outputs"]:
        if Path(name).name != name:
            raise ValueError("Invalid comparison output path")
        outputs.append(directory / name)
    return {
        "script": script,
        "arguments": arguments,
        "implementation": comparison_code_signature(script),
        "runtime": runtime_identity(),
        "entry": entry,
        "contributors": file_hashes(files),
        "fit_inputs": fits,
        "data": source_data_hash(env.DATA_DIR),
    }, outputs


def run_comparison(script: str, arguments: list[str], *, fresh: bool = False) -> bool:
    if (
        Path(script).name != script
        or not (Path(env.ROOT_DIR) / "scripts" / script).is_file()
    ):
        raise ValueError("Name a comparison script in scripts/")
    checkpoint = Path(env.output_root()) / "workflow-checkpoints" / f"{script}.json"
    try:
        inputs, _ = comparison_inputs(script, arguments)
        previous_entry = inputs["entry"]
    except OSError, ValueError, KeyError, TypeError:
        inputs = None
        previous_entry = None
    if not fresh and inputs is not None and checkpoint_matches(checkpoint, inputs):
        print(f"Reusing current comparison: {script}")
        return False
    # A failed attempt must not leave a checkpoint that can stand for success.
    checkpoint.unlink(missing_ok=True)
    subprocess.run(
        [sys.executable, str(Path(env.ROOT_DIR) / "scripts" / script), *arguments],
        cwd=env.ROOT_DIR,
        check=True,
        env={**os.environ, env.OUTPUT_DIR_ENV_VAR: env.output_root()},
    )
    inputs, outputs = comparison_inputs(script, arguments)
    if inputs["entry"] == previous_entry:
        raise RuntimeError(f"{script} did not refresh its comparison manifest entry.")
    record_checkpoint(checkpoint, inputs, outputs)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("script")
    parser.add_argument("--output-dir")
    parser.add_argument("--fresh", action="store_true")
    args, arguments = parser.parse_known_args()
    env.set_output_root(args.output_dir)
    run_comparison(args.script, arguments, fresh=args.fresh)
