# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Filesystem locations for ``vocab_growth`` and the output-root resolver.

The output root is resolved at call time by :func:`output_root`, with precedence:
an explicit override set via :func:`set_output_root` (e.g. from ``--output-dir``) >
the ``DSE_VOCAB_GROWTH_OUTPUT_DIR`` environment variable > the repository-local
``output/`` default. :func:`models_output_dir` / :func:`comparisons_output_dir` are
its ``models`` / ``comparisons`` subdirectories. ``docs/report/figures/``
(``REPORT_FIGS_DIR``) is the report-facing cache and deliberately stays in the
checkout, never under this root.

The resolution *policy* and the disk preflight live in
:mod:`dse_research_utils.environment` (v0.12.0), shared with the other research
repositories. What stays here is this repository's configuration — the
environment-variable name, the repo-local default, and the ``models`` /
``comparisons`` layout — plus ``str``-returning wrappers, since this package's
call sites feed ``os.path.join``.
"""

import os

from dse_research_utils.environment.disk import free_space_gb as _shared_free_space_gb
from dse_research_utils.environment.disk import preflight_disk as _shared_preflight_disk
from dse_research_utils.environment.paths import OutputRoot

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.dirname(_MODULE_DIR)

ROOT_DIR = os.path.dirname(_SRC_DIR)

DATA_DIR = os.path.join(ROOT_DIR, "data")

DOCS_DIR = os.path.join(ROOT_DIR, "docs")

REPORT_DIR = os.path.join(DOCS_DIR, "report")
REPORT_FIGS_DIR = os.path.join(REPORT_DIR, "figures")


# ---------------------------------------------------------------------------
# Output-root resolution
# ---------------------------------------------------------------------------
# Model traces and reporting-quality artefacts are large (a reporting-config
# ``trace.nc`` exceeds 10 GB), so on ephemeral VMs we want to redirect them to a
# scratch disk without disturbing local development or report rendering. The
# output root is therefore resolved at *call time*, with this precedence:
#
#   1. an explicit override set via ``set_output_root`` (e.g. from ``--output-dir``)
#   2. the ``DSE_VOCAB_GROWTH_OUTPUT_DIR`` environment variable
#   3. the repository-local ``<repo>/output`` default (unchanged behaviour)
#
# ``docs/report/figures/`` is deliberately *not* under this root: it is the
# report-facing cache populated by ``scripts/sync_report_figures.py`` and always
# lives in the checkout so the Quarto report renders without the scratch disk.
OUTPUT_DIR_ENV_VAR = "DSE_VOCAB_GROWTH_OUTPUT_DIR"

_DEFAULT_OUTPUT_DIR = os.path.join(ROOT_DIR, "output")

# ``resolve_symlinks=False``: ``<repo>/output`` is a symlink to a scratch volume
# on the fitting VM, and the link path is the stable name recorded in fit
# manifests and blob-upload prefixes, so a configured root is normalised with
# ``expanduser`` + ``abspath`` rather than resolved through the link.
_OUTPUT_ROOT = OutputRoot(
    OUTPUT_DIR_ENV_VAR, _DEFAULT_OUTPUT_DIR, resolve_symlinks=False
)


def set_output_root(path: str | None) -> None:
    """Set a process-wide output-root override (typically from ``--output-dir``).

    Takes precedence over ``$DSE_VOCAB_GROWTH_OUTPUT_DIR``. Pass ``None`` to clear
    the override and fall back to the environment variable / default. Call this
    once, early in a script's entry point, before any output path is resolved.
    """
    _OUTPUT_ROOT.set(path)


def output_root() -> str:
    """Resolve the output root at call time (see module docstring for precedence)."""
    return str(_OUTPUT_ROOT.resolve())


def describe_output_root() -> str:
    """One-line description of the resolved root and its source, for run logs."""
    return _OUTPUT_ROOT.describe()


def models_output_dir() -> str:
    """``<output root>/models`` — one subdirectory per fitted model."""
    return os.path.join(output_root(), "models")


def comparisons_output_dir() -> str:
    """``<output root>/comparisons`` — cross-model / DS-vs-TD comparison artefacts."""
    return os.path.join(output_root(), "comparisons")


# Backwards-compatible module attributes. These were historically constants; they
# now resolve dynamically (PEP 562 module ``__getattr__``) so any remaining
# consumer honours the configured output root. Prefer the ``*_output_dir()``
# helpers (and ``output_root()``) in new code.
def __getattr__(name: str) -> str:
    if name == "OUTPUT_DIR":
        return output_root()
    if name == "MODELS_OUTPUT_DIR":
        return models_output_dir()
    if name == "COMPARISONS_OUTPUT_DIR":
        return comparisons_output_dir()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def free_space_gb(path: str | None = None) -> float:
    """Free space (GiB) on the volume backing ``path`` (default: the output root)."""
    return _shared_free_space_gb(path or output_root())


def preflight_disk(
    min_gb: float, path: str | None = None, *, label: str = "operation"
) -> float:
    """Report free disk space and raise ``RuntimeError`` if below ``min_gb`` (GiB).

    Call at the start of any script that writes large artefacts (model traces are
    >10 GB at reporting configs) so a full volume fails fast rather than after a
    multi-hour sample. Prints the resolved output location so redirected runs are
    obvious in job logs. Returns the free space in GiB when the check passes.
    """
    return _shared_preflight_disk(
        min_gb, path or output_root(), label=label, output_root=output_root()
    )
