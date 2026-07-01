# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Fit prior-sensitivity variants of a model of record (issue #89 §7).

Usage:
    python scripts/fit_sensitivity.py <model> <variant|all> [--config test]

Builds the alternative-prior variant(s) from ``vocab_growth.sensitivity.registry``
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
# engines; VG13 is intentionally not in the registry — too heavy at `test`).
_RUNNER_BY_KEY = {
    "vg10": fit_bivariate_re_model,
    "vg11": fit_univariate_re_model,
    "vg15": fit_joint_model,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="Model key with sensitivity variants (vg10, vg11, vg15).")
    parser.add_argument("variant", help="Variant name, or 'all' for every variant of the model.")
    parser.add_argument(
        "--config",
        default="test",
        help="Sampling configuration (default: test — the honest tier for robustness).",
    )

    freeze_support()
    setup.init_script()
    args = parser.parse_args()

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
        ],
    )
    env.preflight_disk(
        2.0 * len(variant_defs),
        env.OUTPUT_DIR,
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
