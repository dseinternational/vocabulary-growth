# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Summarise registered sensitivity variants against the model of record.

Usage:
    python scripts/compare_sensitivity.py <model> [--variant all|<name>] [--out CSV]

Resolves the baseline fit (``output/models/<id>-<config_name>/``) and each
variant fit (``…-<suffix>/``) from the registry, compares headline quantities
(are they inside the baseline's 89% interval?), writes a per-variant detail CSV
plus a ``robustness_matrix_<model>.csv`` under ``output/comparisons/sensitivity/``,
and prints the matrix.

**Every registered variant gets a row.** A variant that was never fitted, or
whose fit was stopped by the convergence gate and retained under
``output/failed/``, appears with a status and a reason rather than being skipped
with a console note. A matrix that silently omits what it could not assess reads
as coverage it has not got — the requirement recorded in
``notes/202608142000-refit-run-record-and-disk-failure.md`` §5b, after
``vg11 / anchor-broad`` failed the gate and vanished from its own matrix.

**A targeted rerun merges rather than replaces.** ``--variant <name>`` used to
rewrite the whole matrix from that one row, silently dropping every other
variant's verdict. Carried-over rows keep the ``computed_at_utc`` of the run that
produced them, so a stale row is visible as one.
"""

import argparse
import os
from datetime import UTC, datetime

import pandas as pd

from vocab_growth import environment as env
from vocab_growth.analysis_frames import expected_analysis_frame_hash
from vocab_growth.models.definitions import MODEL_REGISTRY
from vocab_growth.models.implementation_identity import implementation_signature
from vocab_growth.reporting import console, dataframe_table, heading
from vocab_growth.sensitivity.compare import (
    compare_dirs,
    coverage_report,
    failed_fit_dir,
    fit_created_at,
    load_comparable,
    pairing_errors,
    required_quantities,
    summarise,
    summarise_absent,
)
from vocab_growth.sensitivity.registry import VARIANTS, build_variant, variants_for

MATRIX_COLUMNS = [
    "variant",
    "status",
    "converged",
    # The soft tier: divergences, low energy BFMI, unassessable parameters and
    # any accepted R-hat exception. `converged` alone reported only the hard
    # tier, so a variant that had not sampled cleanly could still read as
    # robust in this matrix (issue #266 finding 7).
    "caveats",
    "max_rhat",
    "min_ess",
    "baseline_converged",
    "baseline_max_rhat",
    "baseline_min_ess",
    "n_within_ci",
    "n_checked",
    "coverage",
    "quantities_outside_ci",
    "max_abs_delta",
    "verdict",
    "baseline_fit_utc",
    "variant_fit_utc",
    "computed_at_utc",
]


def _model_dir(model_key: str, suffix: str | None = None) -> str:
    d = MODEL_REGISTRY[model_key]
    name = f"{d.model_id}-{d.config_name}" + (f"-{suffix}" if suffix else "")
    return os.path.join(env.models_output_dir(), name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model",
        help=(
            "Model key with registered sensitivity variants: "
            + ", ".join(sorted({key for key, _ in VARIANTS}))
            + "."
        ),
    )
    parser.add_argument("--variant", default="all", help="Variant name or 'all' (default).")
    parser.add_argument("--out", default=None, help="Robustness-matrix CSV path.")
    args = parser.parse_args()

    if args.model not in MODEL_REGISTRY:
        console.print(f"[bold red]Unknown model: {args.model}[/bold red]")
        raise SystemExit(1)

    definition = MODEL_REGISTRY[args.model]
    baseline = _model_dir(args.model)
    if not os.path.isdir(baseline):
        console.print(
            f"[bold red]Baseline fit not found:[/bold red] {baseline}\n"
            "Fit the model of record at the same config before comparing."
        )
        raise SystemExit(1)

    baseline_fit_utc = fit_created_at(baseline)
    now = datetime.now(UTC).isoformat()

    names = variants_for(args.model) if args.variant == "all" else [args.variant]
    detail_dir = os.path.join(env.comparisons_output_dir(), "sensitivity")
    os.makedirs(detail_dir, exist_ok=True)
    failed_root = os.path.join(env.output_root(), "failed")

    # Every variant is checked against the same baseline and the same checkout,
    # so the expensive inputs are computed once rather than once per row: the
    # executable-code signature hashes the whole package, the baseline's
    # prepared-frame hash rebuilds the frame, and the baseline's own summaries
    # are read by both the coverage report and the comparison. VG15 registers
    # thirty variants. Deferred to the first fitted variant so a run that has
    # only "not fitted" rows to report still needs neither the prepared data nor
    # the baseline's output.
    baseline_inputs: dict = {}

    def _baseline_inputs() -> dict:
        if not baseline_inputs:
            try:
                baseline_inputs.update(
                    signature=implementation_signature(),
                    frame_hash=expected_analysis_frame_hash(args.model, definition),
                    summaries=load_comparable(baseline),
                )
            except (OSError, ValueError, KeyError) as exc:
                console.print(
                    f"[bold red]Baseline fit cannot be read:[/bold red] {baseline}\n{exc}"
                )
                raise SystemExit(1) from exc
        return baseline_inputs

    rows: list[dict] = []
    for name in names:
        spec = VARIANTS.get((args.model, name))
        if spec is None:
            console.print(f"[yellow]Unknown variant {name!r}; skipping.[/yellow]")
            continue
        vdir = _model_dir(args.model, spec["suffix"])
        variant_config = f"{definition.config_name}-{spec['suffix']}"

        if not os.path.isdir(vdir):
            # Not in models/ — but it may have been fitted and stopped by the
            # convergence gate, in which case the trace and diagnostics were
            # retained under failed/ and the failure is the finding.
            fdir = failed_fit_dir(failed_root, definition.model_id, variant_config)
            if fdir is not None:
                row = summarise_absent(
                    name,
                    "failed",
                    "FAILED THE CONVERGENCE GATE (not assessed) — retained at "
                    f"failed/{os.path.basename(fdir)}",
                    variant_dir=fdir,
                )
                row["variant_fit_utc"] = fit_created_at(fdir)
            else:
                row = summarise_absent(
                    name, "not-fitted", "NOT FITTED (no comparison available)"
                )
                row["variant_fit_utc"] = None
            row["baseline_fit_utc"] = baseline_fit_utc
            row["computed_at_utc"] = now
            rows.append(row)
            continue

        shared = _baseline_inputs()
        errors = pairing_errors(
            baseline, vdir, args.model, name,
            signature=shared["signature"], baseline_frame_hash=shared["frame_hash"],
        )
        variant_definition, = build_variant(args.model, name)
        required = (required_quantities(args.model, definition)
                    | required_quantities(args.model, variant_definition))
        try:
            summaries = (shared["summaries"], load_comparable(vdir))
            coverage = coverage_report(baseline, vdir, required=required, summaries=summaries)
            comparison = compare_dirs(baseline, vdir, summaries=summaries)
            row = summarise(
                comparison, vdir, label=name, baseline_dir=baseline,
                validation_errors=errors, coverage=coverage,
            )
            # Only a real comparison replaces the detail file. Writing an empty
            # frame on the failure path below would destroy the last good detail
            # for this variant, which is the only per-age record there is.
            comparison.to_csv(
                os.path.join(detail_dir, f"{args.model}-{name}.csv"), index=False
            )
        except (OSError, ValueError, KeyError) as exc:
            # The pairing errors are the more actionable finding when a fit is
            # both stale and unreadable, so they travel with the reason; and the
            # fit directory exists, so its convergence gate is still reported.
            reason = "; ".join([f"NOT ASSESSED: {exc}", *errors])
            row = summarise_absent(name, "invalid-summary", reason, variant_dir=vdir)
        row["baseline_fit_utc"] = baseline_fit_utc
        row["variant_fit_utc"] = fit_created_at(vdir)
        row["computed_at_utc"] = now
        rows.append(row)

    if not rows:
        console.print("[bold red]No registered variants to compare.[/bold red]")
        raise SystemExit(1)

    matrix = pd.DataFrame(rows)
    out = args.out or os.path.join(detail_dir, f"robustness_matrix_{args.model}.csv")

    # A targeted rerun updates its own rows and leaves the rest standing, rather
    # than rewriting the matrix down to the single variant just recomputed.
    if args.variant != "all" and os.path.exists(out):
        previous = pd.read_csv(out)
        if "variant" in previous.columns:
            kept = previous[~previous["variant"].isin(matrix["variant"])]
            matrix = pd.concat([kept, matrix], ignore_index=True)
        order = {name: i for i, name in enumerate(variants_for(args.model))}
        matrix = matrix.sort_values(
            "variant", key=lambda s: s.map(lambda v: order.get(v, len(order)))
        ).reset_index(drop=True)

    for column in MATRIX_COLUMNS:
        if column not in matrix.columns:
            matrix[column] = None
    matrix = matrix[MATRIX_COLUMNS]
    matrix.to_csv(out, index=False)

    heading(f"Sensitivity robustness — {args.model}")
    dataframe_table(
        matrix[
            ["variant", "status", "verdict", "max_abs_delta", "n_within_ci", "n_checked", "max_rhat", "min_ess"]
        ],
        title="Robustness matrix",
        show_index=False,
    )
    assessed = matrix[matrix["status"] == "compared"]
    console.print(
        f"[dim]{len(assessed)} of {len(matrix)} registered variants assessed; "
        f"baseline fitted {baseline_fit_utc}[/dim]"
    )
    console.print(f"[dim]Wrote {out}[/dim]")
