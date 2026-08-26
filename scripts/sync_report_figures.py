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
import csv
import os
import shutil
import tempfile
import uuid
from dataclasses import asdict
from typing import Any

import dse_research_utils.statistics.models.sampling as sampling

from vocab_growth import environment as env
from vocab_growth.analysis_frames import expected_analysis_frame_hash
from vocab_growth.comparisons_provenance import (
    COMPARISON_MANIFEST_FILENAME,
    validate_comparison_manifest,
)
from vocab_growth.fit_artifacts import (
    DIAGNOSTICS_SUMMARY_FILENAME,
    FitValidationError,
    fit_validation_kwargs,
    read_convergence_caveats,
    read_json,
    source_data_hash,
    validate_fit_output,
)
from vocab_growth.models.definitions import MODEL_REGISTRY

COPY_EXTS = (".svg", ".png", ".csv")

CONVERGENCE_CAVEATS_TABLE = "convergence_caveats.csv"
CONVERGENCE_DIAGNOSTICS_TABLE = "convergence_diagnostics.csv"


def _write_csv(filename: str, header: list[str], rows: list[list[Any]]) -> str:
    """Atomically (re)write one generated table into the report figure cache."""
    os.makedirs(env.REPORT_FIGS_DIR, exist_ok=True)
    path = os.path.join(env.REPORT_FIGS_DIR, filename)
    tmp = f"{path}.{uuid.uuid4().hex}.tmp"
    with open(tmp, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
    os.replace(tmp, path)
    return path


def _write_convergence_records(model_sources: list[tuple[str, str]]) -> None:
    """Record each synced model's achieved diagnostics and soft-tier caveats.

    Emitted as CSV rather than left for the report to read from each fit's
    ``diagnostics_summary.json``, because :data:`COPY_EXTS` deliberately syncs only
    figures and summary tables — a JSON reader in the report would silently render
    "pending" forever. Widening the allowlist would instead push the fit manifest
    and lifecycle state into the cache, which the report has no use for.

    Written on every sync, not only under ``--allow-caveats``, and always
    rewritten — including to a header-only file when nothing is caveated. A stale
    table left by an earlier run would otherwise keep asserting caveats against
    fits that no longer carry them, or vanish and let a caveated fit render as
    clean. Appendix B reads these files, so "no file" and "no caveats" must stay
    distinguishable.

    The rendered mark is what makes ``--allow-caveats`` honest rather than a way to
    bypass the check, so this is deliberately not conditional on the flag.
    """
    diagnostics: list[list[Any]] = []
    caveats: list[list[Any]] = []

    # Resolve the model id from the registry rather than by splitting the directory
    # name: sensitivity-variant directories share a model's prefix (e.g.
    # VG10-...-us01-implausible-reinstated), so a prefix split would silently emit
    # two rows for the same model id. Callers only pass registered labels, but the
    # lookup makes that a guarantee rather than an assumption.
    model_id_by_label = {
        f"{d.model_id}-{d.config_name}": d.model_id for d in MODEL_REGISTRY.values()
    }

    for name, src in model_sources:
        model_id = model_id_by_label.get(name)
        if model_id is None:
            continue
        gate = read_json(os.path.join(src, DIAGNOSTICS_SUMMARY_FILENAME)) or {}
        checks = gate.get("checks") or {}
        bfmi = [b for b in (gate.get("bfmi_per_chain") or []) if b is not None]
        soft = [
            label
            for label, ok in (
                ("divergences", checks.get("divergences")),
                ("BFMI", checks.get("bfmi")),
            )
            if ok is False
        ]
        diagnostics.append([
            model_id,
            gate.get("divergences"),
            gate.get("max_rhat"),
            gate.get("min_ess"),
            min(bfmi) if bfmi else None,
            ", ".join(soft),
        ])
        for caveat in read_convergence_caveats(src):
            # Caveats read "<summary>: <consequence>"; keep both, split cleanly.
            summary, _, consequence = caveat.partition(":")
            caveats.append([model_id, summary.strip(), consequence.strip()])

    _write_csv(
        CONVERGENCE_DIAGNOSTICS_TABLE,
        ["model", "divergences", "max_rhat", "min_ess", "min_bfmi", "soft_caveats"],
        sorted(diagnostics),
    )
    _write_csv(
        CONVERGENCE_CAVEATS_TABLE,
        ["model", "caveat", "consequence"],
        sorted(caveats),
    )

    if caveats:
        print(f"  convergence caveats: {len(caveats)} recorded across "
              f"{len({row[0] for row in caveats})} model(s)")
        for model_id, summary, _ in sorted(caveats):
            print(f"    {model_id}: {summary}")
    else:
        print("  convergence caveats: none")


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
    parser.add_argument(
        "--allow-caveats",
        action="store_true",
        help=(
            "Sync reporting-quality fits that cleared the hard convergence tier "
            "but carry soft-tier caveats (divergences, energy BFMI < 0.3). Every "
            "other publication check still applies. The caveats are written to "
            "convergence_caveats.csv in the figure cache and rendered by "
            "Appendix B, so they travel with the numbers."
        ),
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
                # Rebuilt per definition: catches loader-rule drift the raw-CSV
                # fingerprint cannot (issue #266 finding 1). Skipped for
                # provisional syncs, whose kwargs profile carries no data checks.
                current_frame_hash = (
                    None
                    if args.allow_provisional
                    else expected_analysis_frame_hash(
                        definition.model_id.lower(), definition
                    )
                )
                errors = validate_fit_output(
                    src,
                    **fit_validation_kwargs(
                        "provisional-sync"
                        if args.allow_provisional
                        else ("sync-with-caveats" if args.allow_caveats else "sync"),
                        expected_definition=definition,
                        expected_sampling_config_name=args.config,
                        expected_sampling_parameters=asdict(expected_sampling),
                        current_source_data_hash=current_source_hash,
                        current_analysis_frame_hash=current_frame_hash,
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

            _write_convergence_records(model_sources)
        else:
            print(f"[skip] no models output dir: {models_dir}")

    if not args.models_only:
        if os.path.isdir(comparisons_dir):
            # Comparison outputs are derived from fitted output but carried no
            # provenance of their own, so a comparison generated from a
            # since-replaced fit synced as though it were current (issue #266
            # finding 1). Unclaimed files are reported rather than rejected:
            # the manifest is being adopted script by script, and a warning
            # names what is still unrecorded without blocking the rest.
            comparison_errors, comparison_warnings = validate_comparison_manifest(
                comparisons_dir, models_dir
            )
            for warning in comparison_warnings:
                print(f"[warn] {warning}")
            if comparison_errors and not args.allow_provisional:
                for error in comparison_errors:
                    print(f"[invalid] {error}")
                raise FitValidationError(
                    "Comparison outputs failed provenance validation; "
                    f"regenerate them, or see {COMPARISON_MANIFEST_FILENAME}."
                )
            for error in comparison_errors:
                print(f"[provisional] {error}")

            n = _sync_dir(
                comparisons_dir, os.path.join(env.REPORT_FIGS_DIR, "comparisons")
            )
            total += n
            print(f"  comparisons: {n} files")

            # ``_sync_dir`` is deliberately flat, so the nested sub-directories
            # under ``comparisons/`` have to be named. Recovery scores are cited
            # by the report's bias caveats and were invisible to it until this
            # was added; sensitivity is listed for the same reason.
            for sub in ("recovery", "sensitivity"):
                nested = os.path.join(comparisons_dir, sub)
                if not os.path.isdir(nested):
                    continue
                n = _sync_dir(
                    nested, os.path.join(env.REPORT_FIGS_DIR, "comparisons", sub)
                )
                total += n
                print(f"  comparisons/{sub}: {n} files")
        else:
            print(f"[skip] no comparisons dir: {comparisons_dir}")

    print(f"[done] synced {total} files into {env.REPORT_FIGS_DIR}")


if __name__ == "__main__":
    main()
