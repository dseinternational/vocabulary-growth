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
import tempfile
import uuid
from dataclasses import asdict

import dse_research_utils.statistics.models.sampling as sampling

from vocab_growth import environment as env
from vocab_growth.fit_artifacts import (
    FitValidationError,
    fit_validation_kwargs,
    source_data_hash,
    validate_fit_output,
)
from vocab_growth.models.definitions import MODEL_REGISTRY

COPY_EXTS = (".svg", ".png", ".csv")


def _sync_dir(src: str, dst: str) -> int:
    """Replace ``dst`` with an allowlisted snapshot of ``src``."""
    parent = os.path.dirname(dst)
    os.makedirs(parent, exist_ok=True)
    staged = tempfile.mkdtemp(prefix=f".{os.path.basename(dst)}-", dir=parent)
    backup = os.path.join(parent, f".{os.path.basename(dst)}-backup-{uuid.uuid4().hex}")
    copied = 0
    try:
        for name in os.listdir(src):
            source = os.path.join(src, name)
            if os.path.isfile(source) and name.lower().endswith(COPY_EXTS):
                shutil.copy2(source, os.path.join(staged, name))
                copied += 1

        had_destination = os.path.exists(dst)
        if had_destination:
            os.replace(dst, backup)
        try:
            os.replace(staged, dst)
        except BaseException:
            if had_destination and os.path.exists(backup):
                os.replace(backup, dst)
            raise
        if had_destination:
            shutil.rmtree(backup)
    finally:
        if os.path.isdir(staged):
            shutil.rmtree(staged)
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
    parser.add_argument(
        "--config",
        default="rep",
        help="Expected sampling configuration for model artefacts (default: rep).",
    )
    parser.add_argument(
        "--allow-provisional",
        action="store_true",
        help="Allow complete dev/test fits in the local cache.",
    )
    args = parser.parse_args()
    if args.models_only and args.comparisons_only:
        parser.error("Choose --models-only or --comparisons-only, not both.")

    env.set_output_root(args.output_dir)
    models_dir = env.models_output_dir()
    comparisons_dir = env.comparisons_output_dir()
    print(f"[output] reading fitted output from {env.output_root()}")

    total = 0
    definitions_by_label = {
        f"{definition.model_id}-{definition.config_name}": definition
        for definition in MODEL_REGISTRY.values()
    }
    model_sources: list[tuple[str, str]] = []

    if not args.comparisons_only:
        if os.path.isdir(models_dir):
            expected_sampling = sampling.get_sampling_configuration(args.config)
            current_source_hash = (
                None if args.allow_provisional else source_data_hash(env.DATA_DIR)
            )
            validation_failures: list[tuple[str, list[str]]] = []
            for name in sorted(os.listdir(models_dir)):
                src = os.path.join(models_dir, name)
                definition = definitions_by_label.get(name)
                if not os.path.isdir(src):
                    print(f"[skip] non-directory model output: {name}")
                    continue
                if definition is None:
                    print(f"[skip] unregistered model output: {name}")
                    continue
                errors = validate_fit_output(
                    src,
                    **fit_validation_kwargs(
                        "provisional-sync" if args.allow_provisional else "sync",
                        expected_definition=definition,
                        expected_sampling_config_name=args.config,
                        expected_sampling_parameters=asdict(expected_sampling),
                        current_source_data_hash=current_source_hash,
                    ),
                )
                if errors:
                    validation_failures.append((name, errors))
                else:
                    model_sources.append((name, src))

            if validation_failures:
                for name, errors in validation_failures:
                    print(f"[invalid] {name}")
                    for error in errors:
                        print(f"  - {error}")
                raise FitValidationError(
                    "No report figures were changed because one or more model "
                    "outputs failed validation."
                )

            for name, src in model_sources:
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
