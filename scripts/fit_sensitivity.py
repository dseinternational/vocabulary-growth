# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Fit registered robustness variants of a model of record.

Usage:
    python scripts/fit_sensitivity.py <model> <variant|all> [--config test]

Builds the requested variant(s) from ``vocab_growth.sensitivity.registry``
and runs the SAME fit pipeline as the model of record (via its RE / joint runner),
writing output to ``output/models/<model_id>-<config_name>-<suffix>/`` so the
model of record is never touched. Defaults to the ``test`` tier — the honest
config for a robustness claim (``dev``'s short chains under-converge the
hierarchical models).
"""

import argparse
import sys
import time
from multiprocessing import freeze_support

import dse_research_utils.environment.setup as setup

from vocab_growth import environment as env
from vocab_growth.models.catalogue import engine_for
from vocab_growth.models.common import is_reporting_quality_config
from vocab_growth.reporting import (
    console,
    format_duration,
    key_value_table,
    pipeline_summary,
)
from vocab_growth.sensitivity.registry import VARIANTS, build_variant, variants_for

#: Models with registered sensitivity variants, each paired with the engine fit
#: function the catalogue says fits it. A variant is the model of record with
#: one field overridden, so it fits through the model's own engine by
#: construction -- deriving the pairing removes what was a fifth hand-maintained
#: copy of the engine assignment (issue #273). VG13 is intentionally limited to
#: its single-administration variant, which reduces rather than multiplies the
#: full repeated-measures fit cost.
_MODELS_WITH_VARIANTS = sorted({model_key for model_key, _ in VARIANTS})


def _runner(model_key: str):
    """The engine fit function for ``model_key``'s variants."""
    return engine_for(model_key).resolve("fit")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model",
        help=(
            "Model key with sensitivity variants "
            f"({', '.join(_MODELS_WITH_VARIANTS)})."
        ),
    )
    parser.add_argument("variant", help="Variant name, or 'all' for every variant of the model.")
    parser.add_argument(
        "--config",
        default="test",
        help="Sampling configuration (default: test — the honest tier for robustness).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Root directory for model output (overrides "
            "$DSE_VOCAB_GROWTH_OUTPUT_DIR; default: <repo>/output)."
        ),
    )

    freeze_support()
    args = parser.parse_args()
    # Set the output root before init_script(), in case script setup reads a path.
    env.set_output_root(args.output_dir)
    setup.init_script()

    if args.model not in _MODELS_WITH_VARIANTS:
        console.print(
            f"[bold red]No sensitivity variants for model: {args.model}[/bold red] "
            f"(available: {', '.join(_MODELS_WITH_VARIANTS)})"
        )
        sys.exit(1)

    runner = _runner(args.model)
    names = variants_for(args.model) if args.variant == "all" else [args.variant]
    variant_defs = build_variant(args.model, args.variant)

    key_value_table(
        "Sensitivity run plan",
        [
            ("Model", args.model),
            ("Variants", ", ".join(names)),
            ("Sampling config", args.config),
            ("Fits", len(variant_defs)),
            ("Output root", env.output_root()),
        ],
    )
    heavy = is_reporting_quality_config(args.config)
    env.preflight_disk(
        (20.0 if heavy else 2.0) * len(variant_defs),
        env.output_root(),
        label=f"{len(variant_defs)} sensitivity fit(s) [{args.config}]",
    )

    run_started = time.perf_counter()
    timings: dict[str, float] = {}
    for vdef in variant_defs:
        started = time.perf_counter()
        runner(args.config, vdef)
        timings[vdef.config_name] = time.perf_counter() - started

    if len(variant_defs) > 1:
        pipeline_summary("Sensitivity run summary", timings)
    console.print(
        f"[dim]Total wall time: {format_duration(time.perf_counter() - run_started)}[/dim]"
    )
