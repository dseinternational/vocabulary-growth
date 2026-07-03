# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Sync fitted figures and summary tables into the report's figure store.

The Quarto technical report reads its plots and per-model summary tables **only**
from ``docs/report/figures/`` (see ``docs/report/_report_data.qmd``), never from
``output/`` directly, so the report stays renderable from a checkout without the
multi-gigabyte traces. This script refreshes that store from the latest fits:

* ``output/models/<MODEL>-<config>/`` -> ``docs/report/figures/<MODEL>-<config>/``
* ``output/comparisons/``             -> ``docs/report/figures/comparisons/``

Only plots (``.svg``/``.png``) and summary tables (``.csv``) are copied; trace
files (``.nc``) and logs are excluded by the allowlist. ``docs/report/figures/``
is gitignored. Run this after fitting models or regenerating the DS/TD
comparisons, and before rendering the report.

Usage::

    python scripts/sync_report_figures.py                 # models + comparisons
    python scripts/sync_report_figures.py --models-only
    python scripts/sync_report_figures.py --comparisons-only
    python scripts/sync_report_figures.py --output-dir /scratch/vg-output

The source output root follows the same resolution as the fitting scripts:
``--output-dir`` overrides ``$DSE_VOCAB_GROWTH_OUTPUT_DIR``, which overrides the
repository-local ``output/`` default. The report figure store
(``docs/report/figures/``) always stays in the checkout.
"""

from __future__ import annotations

import argparse
import os
import shutil

from vocab_growth import environment as env

COPY_EXTS = (".svg", ".png", ".csv")


def _sync_dir(src: str, dst: str) -> int:
    """Copy allowlisted files from ``src`` into ``dst`` (created if needed)."""
    os.makedirs(dst, exist_ok=True)
    copied = 0
    for name in os.listdir(src):
        s = os.path.join(src, name)
        if os.path.isfile(s) and name.lower().endswith(COPY_EXTS):
            shutil.copy2(s, os.path.join(dst, name))
            copied += 1
    return copied


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sync report figures from output/ into docs/report/figures/."
    )
    parser.add_argument(
        "--models-only", action="store_true", help="Sync only per-model figures."
    )
    parser.add_argument(
        "--comparisons-only",
        action="store_true",
        help="Sync only the DS/TD comparison figures.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Root directory to read fitted output from (overrides "
            "$DSE_VOCAB_GROWTH_OUTPUT_DIR; default: <repo>/output)."
        ),
    )
    args = parser.parse_args()

    env.set_output_root(args.output_dir)
    models_dir = env.models_output_dir()
    comparisons_dir = env.comparisons_output_dir()
    print(f"[output] reading fitted output from {env.output_root()}")

    total = 0

    if not args.comparisons_only:
        if os.path.isdir(models_dir):
            for name in sorted(os.listdir(models_dir)):
                src = os.path.join(models_dir, name)
                if os.path.isdir(src):
                    n = _sync_dir(src, os.path.join(env.REPORT_FIGS_DIR, name))
                    total += n
                    print(f"  {name}: {n} files")
        else:
            print(f"[skip] no models output dir: {models_dir}")

    if not args.models_only:
        if os.path.isdir(comparisons_dir):
            n = _sync_dir(
                comparisons_dir, os.path.join(env.REPORT_FIGS_DIR, "comparisons")
            )
            total += n
            print(f"  comparisons: {n} files")
        else:
            print(f"[skip] no comparisons dir: {comparisons_dir}")

    print(f"[done] synced {total} files into {env.REPORT_FIGS_DIR}")


if __name__ == "__main__":
    main()
