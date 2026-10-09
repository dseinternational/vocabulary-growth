# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Classify CI changes and fingerprint the compiled-cache environment.

Only Ruff, mypy and their exclusively used dependencies are removed from the
lock comparison. Shared dependencies, test tools and unknown paths still run
the full checks. This script uses the standard library so the changes job does
not need to install the scientific environment.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS = {"ruff", "mypy"}
NODE_TOOLS = {"cspell", "prettier", "@cspell/dict-it-it"}


def _closure(packages: list[dict], roots: set[str]) -> set[str]:
    reached = set(roots)
    while True:
        previous = reached.copy()
        for package in packages:
            if package["name"] in reached:
                reached.update(item["name"] for item in package.get("dependencies", []))
                for items in package.get("optional-dependencies", {}).values():
                    reached.update(item["name"] for item in items)
        if reached == previous:
            return reached


def lock_without_tools(lock: dict) -> dict:
    """Retain every lock field except dependencies exclusive to lint tools."""
    lock = copy.deepcopy(lock)
    packages = lock["package"]
    project = next(p for p in packages if p["name"] == "dse-vocab-growth")
    roots = {item["name"] for item in project.get("dependencies", [])}
    for items in project.get("dev-dependencies", {}).values():
        roots.update(item["name"] for item in items if item["name"] not in TOOLS)
    for items in project.get("optional-dependencies", {}).values():
        roots.update(item["name"] for item in items)
    excluded = _closure(packages, TOOLS) - _closure(packages, roots)
    for groups in (
        project.get("dev-dependencies", {}),
        project.get("metadata", {}).get("requires-dev", {}),
    ):
        for name, items in groups.items():
            groups[name] = [item for item in items if item["name"] not in TOOLS]
    lock["package"] = [p for p in packages if p["name"] not in excluded]
    return lock


def _project_without_tool_versions(project: dict) -> dict:
    project = copy.deepcopy(project)
    for name, tool in (("lint", "ruff"), ("typecheck", "mypy")):
        items = project.get("dependency-groups", {}).get(name, [])
        # Other requirements, markers and direct URLs are not a version bump.
        if (
            len(items) != 1
            or not isinstance(items[0], str)
            or not re.fullmatch(
                rf"{tool}\s*(?:[<>=!~]+\s*[\w.*+-]+(?:\s*,\s*[<>=!~]+\s*[\w.*+-]+)*)?",
                items[0],
            )
        ):
            raise ValueError(f"Unrecognised {name} group")
        project["dependency-groups"][name] = [tool]
    return project


def _node_project(project: dict) -> dict:
    project = copy.deepcopy(project)
    if (
        project.get("dependencies")
        or set(project.get("devDependencies", {})) - NODE_TOOLS
    ):
        raise ValueError("Unrecognised Node dependencies")
    project["devDependencies"] = sorted(project.get("devDependencies", {}))
    return project


def needs_full_checks(paths: list[str], before, after) -> bool:
    """Compare parsed dependency contents; a read or parse failure runs tests."""
    try:
        for path in paths:
            if path in {
                "AGENTS.md",
                "CLAUDE.md",
                ".github/copilot-instructions.md",
            } or path.startswith("docs/models/"):
                return True
            if path.startswith(("notes/", "docs/")) or path.endswith(".md"):
                continue
            if path == "uv.lock":
                if lock_without_tools(
                    tomllib.loads(before(path))
                ) != lock_without_tools(tomllib.loads(after(path))):
                    return True
            elif path == "pyproject.toml":
                if _project_without_tool_versions(
                    tomllib.loads(before(path))
                ) != _project_without_tool_versions(tomllib.loads(after(path))):
                    return True
            elif path == "package.json":
                if _node_project(json.loads(before(path))) != _node_project(
                    json.loads(after(path))
                ):
                    return True
            else:
                # pnpm lock and workspace settings need a YAML parser to verify
                # their contents. Keep this job stdlib-only and run full checks.
                return True
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        StopIteration,
        subprocess.CalledProcessError,
    ):
        return True
    return False


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def cache_environment() -> str:
    compiler = subprocess.run(
        ["c++", "--version"], capture_output=True, text=True, check=True
    ).stdout
    return _digest(
        {
            "schema": 1,
            "lock": lock_without_tools(tomllib.loads((ROOT / "uv.lock").read_text())),
            "python": sys.version,
            "platform": platform.platform(),
            "compiler": compiler,
            "settings": {
                name: os.environ.get(name)
                for name in (
                    "PYTENSOR_FLAGS",
                    "NUMBA_CPU_NAME",
                    "NUMBA_CPU_FEATURES",
                    "CFLAGS",
                    "CXXFLAGS",
                    "ImageOS",
                    "ImageVersion",
                )
            },
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("classify", "cache-key"))
    parser.add_argument("--base")
    parser.add_argument("--github-output", required=True)
    args = parser.parse_args()
    if args.operation == "classify":
        try:
            paths = (
                subprocess.check_output(
                    ["git", "diff", "--name-only", "-z", args.base, "HEAD"], cwd=ROOT
                )
                .decode()
                .rstrip("\0")
                .split("\0")
            )

            def before(path):
                return subprocess.check_output(
                    ["git", "show", f"{args.base}:{path}"], cwd=ROOT
                ).decode()

            full = needs_full_checks(
                [path for path in paths if path],
                before,
                lambda path: (ROOT / path).read_text(),
            )
        except OSError, ValueError, subprocess.CalledProcessError:
            full = True
        output = f"code={str(full).lower()}\n"
    else:
        sources = {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for directory in ("src", "scripts", "tests")
            for path in sorted((ROOT / directory).rglob("*.py"))
        }
        output = f"environment={cache_environment()}\nsource={_digest(sources)}\n"
    print(output, end="")
    with open(args.github_output, "a", encoding="utf-8") as handle:
        handle.write(output)


if __name__ == "__main__":
    main()
