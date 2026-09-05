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

Because the scope is that wide, the *breadth* of the check is a separate
decision from *where* it is applied: the purposes that syndicate or extend a
fit ask for it, while re-rendering an existing fit and provisional local syncs
do not (:func:`vocab_growth.fit_artifacts.fit_validation_kwargs`). Editing a
plot helper must not make a completed fit unrenderable.

The signature records the evidence, not just the digest: the per-module hashes
and the library versions travel in the manifest, so a mismatch can be reduced
to the modules and packages that actually moved (:func:`describe_difference`)
rather than sending a reader to refit on an unexplained hash.
"""

from __future__ import annotations

import ast
import hashlib
import json
from importlib import metadata
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]

#: Installed distributions whose version is part of the signature. Everything
#: the numerical path runs through: the sampler and its tensor backend, the
#: array/statistics libraries, the frame library the prepared-data hash is
#: computed over, the diagnostics/LOO library every recorded score comes from,
#: and the shared research utilities.
NUMERICAL_PACKAGES = (
    "pymc",
    "pytensor",
    "numpy",
    "scipy",
    "pandas",
    "arviz",
    "nutpie",
    "dse-research-utils",
)

#: Most differing modules named in a mismatch message before it is truncated.
_MAX_NAMED_SOURCES = 5


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
    """Versioned digest recorded by fits and checked by artefact consumers.

    ``sha256`` is what a comparison turns on; ``sources`` and ``packages`` are
    carried so a mismatch can be localised. Comparing the whole payload and
    comparing the digest are the same test, because the digest is taken over
    the payload.
    """
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
    return {"schema_version": 1, "sha256": digest, **payload}


def matches(recorded, expected: dict) -> bool:
    """Whether ``recorded`` is the same implementation as ``expected``.

    The digest is the whole test: it is taken over the payload, so two
    signatures agreeing on ``sha256`` agree on every module and package.
    """
    return isinstance(recorded, dict) and recorded.get("sha256") == expected.get("sha256")


def describe_difference(recorded, expected: dict) -> str:
    """Name what moved between two signatures, for a validation message.

    A fit is refitted on the strength of this sentence, so it has to separate a
    library bump from a code change: without it the reader is told only that a
    hash differs, and cannot tell a ``numpy`` point release from one edited
    likelihood. Falls back to a plain statement when the recorded signature
    predates the payload being stored.
    """
    if not isinstance(recorded, dict):
        return "no signature is recorded"
    changes = []
    for name in NUMERICAL_PACKAGES:
        was = (recorded.get("packages") or {}).get(name)
        now = (expected.get("packages") or {}).get(name)
        if was != now:
            changes.append(f"{name} {_package_label(was)} -> {_package_label(now)}")
    recorded_sources = recorded.get("sources")
    if isinstance(recorded_sources, dict):
        expected_sources = expected.get("sources") or {}
        moved = sorted(
            path
            for path in set(recorded_sources) | set(expected_sources)
            if recorded_sources.get(path) != expected_sources.get(path)
        )
        if moved:
            shown = ", ".join(moved[:_MAX_NAMED_SOURCES])
            if len(moved) > _MAX_NAMED_SOURCES:
                shown += f", and {len(moved) - _MAX_NAMED_SOURCES} more"
            changes.append(f"{len(moved)} module(s) changed: {shown}")
    if not changes:
        # An older signature carried only the digest, so there is nothing to
        # diff against; say that rather than implying nothing moved.
        return "the recorded signature carries no module or package detail to compare"
    return "; ".join(changes)


def _package_label(entry) -> str:
    if entry is None:
        return "absent"
    commit = entry.get("commit")
    version = entry.get("version")
    return f"{version}@{commit[:12]}" if commit else str(version)
