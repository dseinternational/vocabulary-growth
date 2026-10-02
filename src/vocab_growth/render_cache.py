# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Reuse model renders only when report inputs and generated assets still match."""

from __future__ import annotations

import subprocess
from datetime import UTC, datetime
from pathlib import Path

from vocab_growth import environment as env
from vocab_growth.workflow_cache import file_hashes, runtime_identity

RENDER_CHECKPOINT = ".render-checkpoint.json"
REPORT_SUFFIXES = {
    ".qmd",
    ".yml",
    ".yaml",
    ".css",
    ".scss",
    ".tex",
    ".bib",
    ".csl",
    ".docx",
}


def render_inputs(output_dir: str, model_id: str | None) -> dict:
    output = Path(output_dir)
    sources = [
        p
        for p in Path(env.DOCS_DIR).rglob("*")
        if p.is_file()
        and p.suffix in REPORT_SUFFIXES
        and not {"_freeze", ".quarto"}.intersection(p.parts)
    ]
    inputs = [
        p
        for p in output.iterdir()
        if p.is_file()
        and p.name != RENDER_CHECKPOINT
        and p.suffix in {".qmd", ".csv", ".png", ".svg", ".json"}
    ]
    sources.append(Path(env.ROOT_DIR) / "scripts/fit_model.py")
    trace = output / "trace.nc"
    return {
        "model": model_id,
        "files": file_hashes(sources + inputs),
        "runtime": runtime_identity(),
        "trace": {"size": trace.stat().st_size, "modified": trace.stat().st_mtime_ns}
        if trace.exists()
        else None,
        "quarto": subprocess.check_output(["quarto", "--version"], text=True).strip(),
        "date": datetime.now(UTC).date().isoformat(),
    }


def render_outputs(output_dir: str) -> list[Path]:
    output = Path(output_dir)
    paths = [output / "index.html"]
    for name in ("index_files", "site_libs"):
        paths.extend(p for p in (output / name).rglob("*") if p.is_file())
    return paths
