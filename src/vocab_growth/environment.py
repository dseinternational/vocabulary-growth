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
"""

import os
import shutil

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

_output_root_override: str | None = None


def _normalise(path: str) -> str:
    return os.path.abspath(os.path.expanduser(path))


def set_output_root(path: str | None) -> None:
    """Set a process-wide output-root override (typically from ``--output-dir``).

    Takes precedence over ``$DSE_VOCAB_GROWTH_OUTPUT_DIR``. Pass ``None`` to clear
    the override and fall back to the environment variable / default. Call this
    once, early in a script's entry point, before any output path is resolved.
    """
    global _output_root_override
    _output_root_override = _normalise(path) if path else None


def output_root() -> str:
    """Resolve the output root at call time (see module docstring for precedence)."""
    if _output_root_override is not None:
        return _output_root_override
    env_value = os.environ.get(OUTPUT_DIR_ENV_VAR)
    if env_value:
        return _normalise(env_value)
    return _DEFAULT_OUTPUT_DIR


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
    """Free space (GiB) on the volume backing ``path`` (default: the output root).

    Walks up to the nearest existing parent if ``path`` does not exist yet, so it
    works for an output directory that is about to be created.
    """
    target = _normalise(path or output_root())
    while not os.path.exists(target):
        parent = os.path.dirname(target)
        if parent == target:
            break
        target = parent
    return shutil.disk_usage(target).free / (1024 ** 3)


def preflight_disk(
    min_gb: float, path: str | None = None, *, label: str = "operation"
) -> float:
    """Report free disk space and raise ``RuntimeError`` if below ``min_gb`` (GiB).

    Call at the start of any script that writes large artefacts (model traces are
    >10 GB at reporting configs) so a full volume fails fast rather than after a
    multi-hour sample. Prints the resolved output location so redirected runs are
    obvious in job logs. Returns the free space in GiB when the check passes.
    """
    target = _normalise(path or output_root())
    free = free_space_gb(target)
    drive = os.path.splitdrive(target)[0] or target
    # Surface the resolved root only when that is what we are actually checking, so
    # this line can't disagree with the [disk] target when a caller passes a subdir.
    if target == _normalise(output_root()):
        print(f"[output] resolved output root: {output_root()}", flush=True)
    print(f"[disk] {free:.1f} GiB free on {drive} "
          f"(need >= {min_gb:.0f} GiB for {label})", flush=True)
    if free < min_gb:
        raise RuntimeError(
            f"Insufficient disk space for {label}: {free:.1f} GiB free on {drive}, "
            f"need >= {min_gb:.0f} GiB. Free space and retry."
        )
    return free
