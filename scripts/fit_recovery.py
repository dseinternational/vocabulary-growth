# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Run parameter-recovery replicates for a model (issue #163).

Usage:
    python scripts/fit_recovery.py <model|headline|all> [options]

Each replicate does three things: simulate a dataset from the model at a known
parameter draw, refit the model to that dataset with the engine's own pipeline,
and score the recovered posterior against the truth. The stages are separable so
a long run can be resumed or inspected between them:

    --simulate-only     simulate and stop (cheap; no MCMC)
    --fit-only          refit existing simulated data
    --compare-only      score existing recovery fits

The truth comes from the model of record's posterior by default, which asks
whether the parameters are recoverable *in the regime the study reports*. That
requires the model of record to have been fitted at the same output root. Use
``--truth prior`` when no fit exists; see docs/runbooks/parameter-recovery.md for
what each choice does and does not establish.

Recovery fits land in ``<output root>/models/<model_id>-<config>-recovery-rNN/``
and never touch a model of record.
"""

import argparse
import os
import sys
import time
from multiprocessing import freeze_support

import dse_research_utils.environment.setup as setup
import pandas as pd

from vocab_growth import environment as env
from vocab_growth.models.common import is_reporting_quality_config
from vocab_growth.models.definitions import MODEL_REGISTRY
from vocab_growth.recovery.compare import compare_replicate, pooled_row
from vocab_growth.recovery.refit import fit_recovery_replicate, recovery_fit_dir
from vocab_growth.recovery.simulate import (
    available_replicates,
    load_simulation,
    simulate_replicate,
    simulation_dir,
)
from vocab_growth.recovery.spec import HEADLINE_MODELS, supported_models
from vocab_growth.reporting import (
    console,
    dataframe_table,
    format_duration,
    key_value_table,
    pipeline_summary,
)
from vocab_growth.sensitivity.registry import build_variant

RECOVERY_COMPARISON_SUBDIR = os.path.join("comparisons", "recovery")


def _resolve_models(selector: str) -> list[str]:
    if selector == "headline":
        return list(HEADLINE_MODELS)
    if selector == "all":
        return supported_models()
    return [selector]


def _comparison_dir() -> str:
    path = os.path.join(env.output_root(), RECOVERY_COMPARISON_SUBDIR)
    os.makedirs(path, exist_ok=True)
    return path


def _resolve_definition(model_key: str, variant: str | None):
    """The definition to recover: the model of record, or a registered variant."""
    if variant is None:
        return MODEL_REGISTRY[model_key], model_key
    definition = build_variant(model_key, variant)[0]
    return definition, f"{model_key}-{variant}"


def _score(model_key: str, definition, label: str) -> None:
    """Score every simulated replicate of one model and write its recovery matrix.

    Deliberately scores every replicate that exists rather than only the ones the
    current invocation ran: the matrix is the model's whole recovery record, and
    re-scoring a single replicate of a staged run must not drop the others.
    """
    query_ages = pd.Series(definition.ages_query).to_numpy()
    out_dir = _comparison_dir()
    summaries: list[dict] = []
    replicates = available_replicates(definition)
    if not replicates:
        console.print(f"[yellow]{label}: no simulated replicates found.[/yellow]")
        return

    for replicate in replicates:
        sim_dir = simulation_dir(definition, replicate)
        fit_dir = recovery_fit_dir(model_key, replicate, definition=definition)
        if not os.path.isdir(sim_dir):
            console.print(f"[yellow]skip r{replicate:02d}: no simulation at {sim_dir}[/yellow]")
            continue
        trace_path = os.path.join(fit_dir, "trace.nc")
        if not os.path.isfile(trace_path):
            console.print(f"[yellow]skip r{replicate:02d}: no recovery trace at {trace_path}[/yellow]")
            continue
        _frame, truth, record = load_simulation(sim_dir, expected_definition=definition)
        table, aggregates, summary = compare_replicate(
            truth,
            trace_path,
            fit_dir,
            label=f"r{replicate:02d}",
            truth_source=record["simulation"]["truth_source"],
            query_ages=query_ages,
        )
        prefix = f"recovery_{label}_r{replicate:02d}"
        table.to_csv(os.path.join(out_dir, f"{prefix}.csv"), index=False)
        if len(aggregates):
            aggregates.to_csv(os.path.join(out_dir, f"{prefix}_aggregates.csv"), index=False)
            dataframe_table(aggregates, title=f"{label} r{replicate:02d} — random-effect recovery")
        summaries.append(summary)

        worst = table.reindex(table["z"].abs().sort_values(ascending=False).index).head(8)
        dataframe_table(
            worst[
                [
                    "quantity",
                    "index",
                    "truth",
                    "posterior_median",
                    "posterior_sd",
                    "z",
                    "within_ci89",
                ]
            ],
            title=f"{label} r{replicate:02d} — largest standardised errors",
        )

    if not summaries:
        console.print(f"[yellow]{label}: nothing to score.[/yellow]")
        return

    matrix = pd.DataFrame([*summaries, pooled_row(summaries)])
    matrix_path = os.path.join(out_dir, f"recovery_matrix_{label}.csv")
    matrix.to_csv(matrix_path, index=False)
    dataframe_table(
        matrix[
            [
                "replicate",
                "truth_source",
                "converged",
                "n_targets",
                "coverage_ci89",
                "max_abs_z",
                "verdict",
            ]
        ],
        title=f"Parameter recovery — {definition.model_id} [{label}]",
    )
    console.print(f"[dim]Recovery matrix: {matrix_path}[/dim]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model",
        help=(
            "Model key, 'headline' for the three models issue #163 gates "
            f"({', '.join(HEADLINE_MODELS)}), or 'all' for every supported model."
        ),
    )
    parser.add_argument(
        "--variant",
        default=None,
        help=(
            "Recover a registered sensitivity variant instead of the model of "
            "record (e.g. a1-tau-age-varying). Requires a single model, and with "
            "--truth posterior requires that variant to have been fitted."
        ),
    )
    parser.add_argument(
        "--config",
        default="test",
        help="Sampling configuration (default: test — the honest tier for a recovery claim).",
    )
    parser.add_argument(
        "--replicates",
        type=int,
        default=3,
        help="Number of replicates per model (default: 3).",
    )
    parser.add_argument(
        "--replicate",
        type=int,
        action="append",
        default=None,
        help="Run only this replicate number (repeatable). Overrides --replicates.",
    )
    parser.add_argument(
        "--truth",
        choices=("posterior", "prior"),
        default="posterior",
        help=(
            "Where each truth comes from: the model of record's posterior (default; "
            "requires a fitted model of record) or the model's prior."
        ),
    )
    parser.add_argument(
        "--n-prior-draws",
        type=int,
        default=64,
        help="Prior draws to select truths from when --truth prior (default: 64).",
    )
    parser.add_argument(
        "--random-seed", type=int, default=20260725, help="Base simulation seed."
    )
    parser.add_argument("--simulate-only", action="store_true", help="Simulate and stop.")
    parser.add_argument("--fit-only", action="store_true", help="Refit existing simulated data.")
    parser.add_argument("--compare-only", action="store_true", help="Score existing recovery fits.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Root directory for output (overrides $DSE_VOCAB_GROWTH_OUTPUT_DIR; "
            "default: <repo>/output)."
        ),
    )

    freeze_support()
    args = parser.parse_args()
    env.set_output_root(args.output_dir)
    setup.init_script()

    exclusive = [args.simulate_only, args.fit_only, args.compare_only]
    if sum(bool(flag) for flag in exclusive) > 1:
        console.print(
            "[bold red]--simulate-only, --fit-only and --compare-only are mutually "
            "exclusive.[/bold red]"
        )
        sys.exit(1)

    try:
        models = _resolve_models(args.model)
    except KeyError as exc:
        console.print(f"[bold red]{exc}[/bold red]")
        sys.exit(1)

    if args.variant is not None and args.model in {"headline", "all"}:
        console.print(
            "[bold red]--variant applies to one model; name it explicitly.[/bold red]"
        )
        sys.exit(1)
    try:
        definitions = {key: _resolve_definition(key, args.variant) for key in models}
    except KeyError as exc:
        console.print(f"[bold red]{exc}[/bold red]")
        sys.exit(1)

    replicates = (
        sorted(set(args.replicate)) if args.replicate else list(range(1, args.replicates + 1))
    )
    do_simulate = not (args.fit_only or args.compare_only)
    do_fit = not (args.simulate_only or args.compare_only)
    do_compare = not (args.simulate_only or args.fit_only)

    key_value_table(
        "Parameter-recovery run plan",
        [
            ("Models", ", ".join(label for _, label in definitions.values())),
            ("Replicates", ", ".join(f"r{r:02d}" for r in replicates)),
            ("Truth source", args.truth),
            ("Sampling config", args.config),
            ("Stages", ", ".join(
                name
                for name, active in (
                    ("simulate", do_simulate), ("fit", do_fit), ("compare", do_compare)
                )
                if active
            )),
            ("Output root", env.output_root()),
        ],
    )

    if do_fit:
        heavy = is_reporting_quality_config(args.config)
        env.preflight_disk(
            (20.0 if heavy else 2.0) * len(models) * len(replicates),
            env.output_root(),
            label=f"{len(models) * len(replicates)} recovery fit(s) [{args.config}]",
        )

    run_started = time.perf_counter()
    timings: dict[str, float] = {}
    failures: dict[str, str] = {}

    for model_key in models:
        definition, model_label = definitions[model_key]
        for replicate in replicates:
            label = f"{model_label} r{replicate:02d}"
            started = time.perf_counter()
            try:
                if do_simulate:
                    simulate_replicate(
                        model_key,
                        args.config,
                        replicate=replicate,
                        truth_source=args.truth,
                        n_prior_draws=args.n_prior_draws,
                        random_seed=args.random_seed,
                        definition=definition,
                    )
                if do_fit:
                    fit_recovery_replicate(
                        model_key,
                        args.config,
                        replicate=replicate,
                        definition=definition,
                    )
            except Exception as exc:  # one replicate must not sink the run
                failures[label] = f"{type(exc).__name__}: {exc}"
                console.print(f"[bold red]{label} failed:[/bold red] {exc}")
            timings[label] = time.perf_counter() - started

        if do_compare:
            try:
                _score(model_key, definition, model_label)
            except Exception as exc:
                failures[f"{model_label} compare"] = f"{type(exc).__name__}: {exc}"
                console.print(f"[bold red]{model_label} scoring failed:[/bold red] {exc}")

    if len(timings) > 1:
        pipeline_summary("Recovery run summary", timings)
    console.print(
        f"[dim]Total wall time: {format_duration(time.perf_counter() - run_started)}[/dim]"
    )
    if failures:
        key_value_table("Failures", sorted(failures.items()))
        sys.exit(1)
