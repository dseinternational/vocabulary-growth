# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Summarise prior-sensitivity variants against the model of record (issue #89 §7).

Usage:
    python scripts/compare_sensitivity.py <model> [--variant all|<name>] [--out CSV]

Resolves the baseline fit (``output/models/<id>-<config_name>/``) and each
variant fit (``…-<suffix>/``) from the registry, compares headline quantities
(are they inside the baseline's 90% HDI?), writes a per-variant detail CSV plus a
``robustness_matrix_<model>.csv`` under ``output/comparisons/sensitivity/``, and
prints the matrix. Variant fits that are missing are skipped with a note.
"""

import argparse
import os

import pandas as pd

from vocab_growth import environment as env
from vocab_growth.models.definitions import MODEL_REGISTRY
from vocab_growth.reporting import console, dataframe_table, heading
from vocab_growth.sensitivity.compare import compare_dirs, summarise
from vocab_growth.sensitivity.registry import VARIANTS, variants_for


def _model_dir(model_key: str, suffix: str | None = None) -> str:
    d = MODEL_REGISTRY[model_key]
    name = f"{d.model_id}-{d.config_name}" + (f"-{suffix}" if suffix else "")
    return os.path.join(env.models_output_dir(), name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="Model key (vg10, vg11, vg15).")
    parser.add_argument("--variant", default="all", help="Variant name or 'all' (default).")
    parser.add_argument("--out", default=None, help="Robustness-matrix CSV path.")
    args = parser.parse_args()

    if args.model not in MODEL_REGISTRY:
        console.print(f"[bold red]Unknown model: {args.model}[/bold red]")
        raise SystemExit(1)

    baseline = _model_dir(args.model)
    if not os.path.isdir(baseline):
        console.print(
            f"[bold red]Baseline fit not found:[/bold red] {baseline}\n"
            "Fit the model of record at the same config before comparing."
        )
        raise SystemExit(1)

    names = variants_for(args.model) if args.variant == "all" else [args.variant]
    detail_dir = os.path.join(env.comparisons_output_dir(), "sensitivity")
    os.makedirs(detail_dir, exist_ok=True)

    rows: list[dict] = []
    for name in names:
        spec = VARIANTS.get((args.model, name))
        if spec is None:
            console.print(f"[yellow]Unknown variant {name!r}; skipping.[/yellow]")
            continue
        vdir = _model_dir(args.model, spec["suffix"])
        if not os.path.isdir(vdir):
            console.print(f"[yellow]Variant fit not found (skip): {os.path.basename(vdir)}[/yellow]")
            continue
        comparison = compare_dirs(baseline, vdir)
        comparison.to_csv(os.path.join(detail_dir, f"{args.model}-{name}.csv"), index=False)
        rows.append(summarise(comparison, vdir, label=name))

    if not rows:
        console.print("[bold red]No variant fits found to compare.[/bold red]")
        raise SystemExit(1)

    matrix = pd.DataFrame(rows)
    out = args.out or os.path.join(detail_dir, f"robustness_matrix_{args.model}.csv")
    matrix.to_csv(out, index=False)

    heading(f"Prior-sensitivity robustness — {args.model}")
    dataframe_table(
        matrix[
            ["variant", "verdict", "max_abs_delta", "n_within_hdi", "n_checked", "max_rhat", "min_ess"]
        ],
        title="Robustness matrix",
        show_index=False,
    )
    console.print(f"[dim]Wrote {out}[/dim]")
