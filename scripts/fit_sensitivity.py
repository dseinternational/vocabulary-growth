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
from vocab_growth.models.common import is_reporting_quality_config
from vocab_growth.models.common_bivariate_re import fit_bivariate_re_model
from vocab_growth.models.common_joint_modality import fit_joint_model
from vocab_growth.models.common_univariate_re import fit_univariate_re_model
from vocab_growth.reporting import (
    console,
    format_duration,
    key_value_table,
    pipeline_summary,
)
from vocab_growth.sensitivity.registry import build_variant, variants_for

# The sensitivity models of record and the runner each uses (all are RE / joint
# engines). VG13 is intentionally limited to its single-administration variant,
# which reduces rather than multiplies the full repeated-measures fit cost.
_RUNNER_BY_KEY = {
    "vg10": fit_bivariate_re_model,
    "vg11": fit_univariate_re_model,
    "vg12": fit_univariate_re_model,
    "vg13": fit_bivariate_re_model,
    "vg15": fit_joint_model,
    # VG20 is the DS joint model of record and shares VG10's engine; its
    # correlated subject block is a field on the definition, not a different
    # runner. Added 2026-08-19 with the kappa placement variants (#229) -- the
    # first sensitivity variants registered against a model of record.
    "vg20": fit_bivariate_re_model,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model",
        help="Model key with sensitivity variants (vg10, vg11, vg12, vg13, vg15, vg20).",
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

    if args.model not in _RUNNER_BY_KEY:
        console.print(
            f"[bold red]No sensitivity variants for model: {args.model}[/bold red] "
            f"(available: {', '.join(_RUNNER_BY_KEY)})"
        )
        sys.exit(1)

    runner = _RUNNER_BY_KEY[args.model]
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
