# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Fingerprint executable package code independently of Git history and prose.

The scope is deliberately conservative: all Python modules in ``vocab_growth``,
including reporting code, plus the numerical libraries' installed versions.
This avoids a hand-maintained dependency list that can miss a new helper.
Comments, docstrings, whitespace and external documents do not affect the
signature. Other code changes can invalidate more fits than strictly necessary;
accepting a stale fit is the more consequential error. A missing signature is
unverifiable, not an assertion that historical code matches the current code.
"""

from __future__ import annotations

import ast
import hashlib
import json
from importlib import metadata
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
NUMERICAL_PACKAGES = ("pymc", "pytensor", "numpy", "scipy", "nutpie", "dse-research-utils")


class _WithoutDocstrings(ast.NodeTransformer):
    def visit_Expr(self, node):
        # Also covers attribute documentation after a dataclass field, which
        # Python parses as a standalone string rather than a formal docstring.
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            return None
        return self.generic_visit(node)


def executable_source(source: str) -> str:
    """Canonical AST, without source locations or documentation strings."""
    return ast.dump(_WithoutDocstrings().visit(ast.parse(source)), include_attributes=False)


def implementation_signature() -> dict:
    """Versioned digest recorded by fits and checked by artefact consumers."""
    sources = {
        path.relative_to(PACKAGE_ROOT).as_posix(): hashlib.sha256(
            executable_source(path.read_text(encoding="utf-8")).encode("utf-8")
        ).hexdigest()
        for path in sorted(PACKAGE_ROOT.rglob("*.py"))
    }
    packages = {}
    for name in NUMERICAL_PACKAGES:
        try:
            distribution = metadata.distribution(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
        else:
            origin = json.loads(distribution.read_text("direct_url.json") or "{}")
            packages[name] = {
                "version": distribution.version,
                "commit": origin.get("vcs_info", {}).get("commit_id"),
            }
    payload = {"schema_version": 1, "sources": sources, "packages": packages}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return {"schema_version": 1, "sha256": digest}
