# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

import os
import shutil

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.dirname(_MODULE_DIR)

ROOT_DIR = os.path.dirname(_SRC_DIR)

DATA_DIR = os.path.join(ROOT_DIR, "data")

OUTPUT_DIR = os.path.join(ROOT_DIR, "output")

MODELS_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "models")

DOCS_DIR = os.path.join(ROOT_DIR, "docs")

REPORT_DIR = os.path.join(DOCS_DIR, "report")
REPORT_FIGS_DIR = os.path.join(REPORT_DIR, "figures")


def free_space_gb(path: str | None = None) -> float:
    """Free space (GiB) on the volume backing ``path`` (default ``OUTPUT_DIR``).

    Walks up to the nearest existing parent if ``path`` does not exist yet, so it
    works for an output directory that is about to be created.
    """
    target = os.path.abspath(path or OUTPUT_DIR)
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
    multi-hour sample. Returns the free space in GiB when the check passes.
    """
    target = os.path.abspath(path or OUTPUT_DIR)
    free = free_space_gb(target)
    drive = os.path.splitdrive(target)[0] or target
    print(f"[disk] {free:.1f} GiB free on {drive} "
          f"(need >= {min_gb:.0f} GiB for {label})", flush=True)
    if free < min_gb:
        raise RuntimeError(
            f"Insufficient disk space for {label}: {free:.1f} GiB free on {drive}, "
            f"need >= {min_gb:.0f} GiB. Free space and retry."
        )
    return free
