# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Render the report book or comparison page with checked input/output reuse."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from vocab_growth import environment as env
from vocab_growth.fit_artifacts import source_data_hash
from vocab_growth.render_cache import REPORT_SUFFIXES
from vocab_growth.workflow_cache import (
    checkpoint_matches,
    file_hashes,
    record_checkpoint,
    runtime_identity,
)


def report_inputs() -> dict:
    sources = [
        p
        for p in Path(env.DOCS_DIR).rglob("*")
        if p.is_file()
        and p.suffix in REPORT_SUFFIXES | {".csv", ".png", ".svg", ".json"}
        and not {"_freeze", ".quarto", "site_libs", "index_files"}.intersection(p.parts)
    ]
    scripts = [
        Path(env.ROOT_DIR) / "scripts" / name
        for name in ("render_reports.py", "restore_docx_math_settings.py")
    ]
    return {
        "files": file_hashes(sources + scripts),
        "execution": {
            "files": file_hashes(
                [
                    p
                    for p in sources
                    if p.suffix in {".csv", ".json"}
                    or (p.suffix == ".qmd" and p.name.startswith("_"))
                ]
            ),
            "runtime": runtime_identity(),
            "data": source_data_hash(env.DATA_DIR),
        },
        "quarto": subprocess.check_output(["quarto", "--version"], text=True).strip(),
        "date": datetime.now(UTC).date().isoformat(),
    }


def invalidate_frozen_execution(
    checkpoint: Path, inputs: dict, project_dir: Path, *, force: bool
) -> None:
    try:
        previous = json.loads(checkpoint.read_text())["inputs"]["execution"]
    except OSError, ValueError, KeyError, TypeError:
        previous = None
    if force or previous != inputs["execution"]:
        # Only generated Quarto execution caches are removed. Source, figures
        # and fits remain inputs. Plain chapter edits still use freeze: auto.
        shutil.rmtree(project_dir / "_freeze", ignore_errors=True)


def render_report(target: str, *, force: bool = False) -> bool:
    root = Path(env.ROOT_DIR)
    if target not in {"docs/report", "docs/comparison/index.qmd"}:
        raise ValueError("Select docs/report or docs/comparison/index.qmd")
    project = root / target
    directory = project if project.is_dir() else project.parent
    checkpoint = (
        Path(env.output_root())
        / "workflow-checkpoints"
        / f"{directory.name}-render.json"
    )
    inputs = report_inputs()
    if not force and checkpoint_matches(checkpoint, inputs):
        print(f"Reusing current report: {target}")
        return False
    invalidate_frozen_execution(checkpoint, inputs, directory, force=force)
    checkpoint.unlink(missing_ok=True)
    inspected = json.loads(
        subprocess.check_output(["quarto", "inspect", str(project)], text=True)
    )
    config = inspected.get("config", {})
    if project.is_dir():
        output = (directory / config["project"]["output-dir"]).resolve()
        required = [
            output / f"{Path(p).stem}.html"
            for p in inspected["files"]["input"]
            if not Path(p).name.startswith("_")
        ]
    else:
        output = directory
        required = [project.with_suffix(".html")]
    subprocess.run(
        ["quarto", "render", str(project)],
        cwd=root,
        check=True,
        env={**os.environ, "QUARTO_PYTHON": sys.executable},
    )
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise RuntimeError(
            "Quarto did not produce required outputs: " + ", ".join(missing)
        )
    assets = [
        p
        for p in output.rglob("*")
        if p.is_file()
        and not {"_freeze", ".quarto"}.intersection(p.parts)
        and p.suffix not in REPORT_SUFFIXES - {".css", ".scss", ".docx"}
    ]
    record_checkpoint(checkpoint, inputs, required + assets)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", choices=("docs/report", "docs/comparison/index.qmd"))
    parser.add_argument("--output-dir")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    env.set_output_root(args.output_dir)
    render_report(args.target, force=args.force)
