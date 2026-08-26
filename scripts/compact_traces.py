# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Apply a trace-persistence tier to a trace that was already written in full.

``--trace-persistence`` chooses a tier when a fit *writes* its trace. This
applies the same policy afterwards, to fits that were written before the tier
was chosen or under ``full`` by default. It is the recovery path for a full
output volume, which is how it came to exist: on 2026-08-14 the reporting run
filled a 433 GB disk and lost five in-flight refits to ``ENOSPC``.

It reuses :mod:`vocab_growth.fit_artifacts` rather than reimplementing the
policy, so what is dropped here is exactly what a ``--trace-persistence compact``
fit would never have written, and the manifest record is the same shape.

    python scripts/compact_traces.py --dry-run
    python scripts/compact_traces.py VG03-age-spoken-td --tier compact

**What this costs.** ``compact`` keeps every free parameter, ``sample_stats``,
``log_likelihood`` and ``posterior_predictive``, so the reporting output, the
publication gate, ``loo_compare.py`` and the DS/TD comparison suite are all
unaffected. It drops observation-sized deterministics that are recomputable
from the free parameters, and three consumers read those directly and will
refuse a compacted fit up front: ``regenerate_plots.py``, ``loso_compare.py``
and parameter-recovery scoring. Each then needs a refit. Choose accordingly —
in particular, think twice before compacting a model in the recovery headline
set (VG10, VG12, VG15).

See ``notes/202608081445-trace-persistence-tiers.md``.
"""

from __future__ import annotations

import argparse
import os
import sys

import psutil

from vocab_growth import environment as env
from vocab_growth.fit_artifacts import (
    TRACE_FILENAME,
    TracePersistence,
    plan_trace_persistence,
    read_trace_persistence_record,
    record_trace_persistence,
)
from vocab_growth.reporting import console, dataframe_table

STAGING_DIRNAME = ".staging"


def _gib(path: str) -> float:
    return os.path.getsize(path) / 1024**3


def _model_dirs(root: str, names: list[str]) -> list[str]:
    models = os.path.join(root, "models")
    if names:
        return [os.path.join(models, name) for name in names]
    found = [
        os.path.join(models, name)
        for name in os.listdir(models)
        if os.path.isfile(os.path.join(models, name, TRACE_FILENAME))
    ]
    # Smallest first. Each rewrite needs room for its output *beside* the
    # original, so on a nearly-full volume the order is what decides whether the
    # largest trace can be rewritten at all: the small ones free the space it
    # needs. Alphabetical order happens to work today and would not survive a
    # renamed model.
    return sorted(found, key=lambda d: os.path.getsize(os.path.join(d, TRACE_FILENAME)))


def _is_live(staging_entry: str) -> bool:
    """Whether a staging directory belongs to a process that still exists.

    Staging names end ``-<timestamp>-<pid>-<hash>``. A crashed or ``ENOSPC``-killed
    fit leaves its directory behind, and this script exists precisely for the
    aftermath of such a run — so "staging exists" cannot mean "a fit is running".
    An unreadable or unparseable name is treated as live: the conservative
    direction is to refuse. The probe is ``psutil.pid_exists`` because it works
    on every platform this project supports — ``os.kill(pid, 0)`` is not a
    portable liveness check, since CPython's ``os.kill`` on Windows calls
    ``TerminateProcess`` — and a probe that fails for any reason is likewise
    treated as live.
    """
    parts = staging_entry.rsplit("-", 3)
    if len(parts) < 4 or not parts[2].isdigit():
        return True
    try:
        return psutil.pid_exists(int(parts[2]))
    except Exception:
        return True


def _current_tier(directory: str) -> str:
    record = read_trace_persistence_record(directory)
    # Fits written before the setting existed carry no record and are `full`,
    # which is the same convention `fit_artifacts` documents.
    return (record or {}).get("persistence", "full")


def compact_one(directory: str, tier: TracePersistence, *, dry_run: bool) -> dict:
    """Rewrite one fit's trace at ``tier``. Returns a row for the summary table."""
    import xarray as xr

    path = os.path.join(directory, TRACE_FILENAME)
    name = os.path.basename(directory)
    before = _gib(path)
    row = {"model": name, "GiB before": round(before, 2), "GiB after": None,
           "dropped": 0, "status": ""}

    existing = _current_tier(directory)
    if existing != TracePersistence.FULL.value:
        row["status"] = f"skipped (already {existing})"
        return row

    trace = xr.open_datatree(path)
    try:
        plan = plan_trace_persistence(trace, tier)
        if not plan:
            row["status"] = "skipped (nothing droppable)"
            return row
        row["dropped"] = sum(len(names) for names in plan.values())
        if dry_run:
            row["status"] = "would rewrite"
            return row

        # Write beside the original, then swap: a half-written trace must never
        # be able to take the place of a complete one, and this project has
        # already lost fits to a truncated write.
        tmp = path + ".compacting"
        from vocab_growth.fit_artifacts import _filtered_trace

        _filtered_trace(trace, plan).to_netcdf(tmp)
    finally:
        trace.close()

    # Verify the replacement before it replaces anything: it must open, and it
    # must still carry every free parameter the original had. A tier that
    # silently dropped a sampled variable would be indistinguishable from a
    # corrupt file later.
    with xr.open_datatree(path) as original, xr.open_datatree(tmp) as rewritten:
        kept = set(rewritten["posterior"].to_dataset().data_vars)
        expected = set(original["posterior"].to_dataset().data_vars) - set(
            plan.get("posterior", [])
        )
        missing = expected - kept
        if missing:
            os.remove(tmp)
            raise RuntimeError(
                f"{name}: rewrite lost {sorted(missing)[:5]} — original left intact."
            )

    os.replace(tmp, path)
    record_trace_persistence(
        directory,
        {
            "persistence": tier.value,
            "dropped": plan,
            "dropped_count": row["dropped"],
            "applied_after_fit_by": "scripts/compact_traces.py",
        },
    )
    row["GiB after"] = round(_gib(path), 2)
    row["status"] = "rewritten"
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", nargs="*", help="Output directory names (default: all).")
    parser.add_argument(
        "--tier",
        choices=[t.value for t in TracePersistence if t is not TracePersistence.FULL],
        default=TracePersistence.COMPACT.value,
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help=(
            "Output directory name to leave at `full` (repeatable). Use it for "
            "models whose recovery, LOSO or plot regeneration is still to run."
        ),
    )
    parser.add_argument("--dry-run", action="store_true", help="Report and change nothing.")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    env.set_output_root(args.output_dir)
    root = env.output_root()

    tier = TracePersistence(args.tier)
    excluded = set(args.exclude)
    directories = [
        d for d in _model_dirs(root, args.models) if os.path.basename(d) not in excluded
    ]
    if excluded:
        console.print(f"[dim]Leaving at full: {', '.join(sorted(excluded))}[/dim]")

    # Refuse only where a staging entry belongs to a model being rewritten. A
    # blanket "staging is non-empty" refusal is wrong in the situation this
    # script exists for: the run that fills the disk leaves stale staging
    # directories behind, and a *live* fit of some other model is exactly what
    # the free space is being reclaimed for. Staging names are
    # `<config_name>-<timestamp>-<pid>-<hash>`.
    staging = os.path.join(root, STAGING_DIRNAME)
    if os.path.isdir(staging):
        entries = os.listdir(staging)
        for directory in directories:
            name = os.path.basename(directory)
            racing = [e for e in entries if e.startswith(f"{name}-") and _is_live(e)]
            if racing:
                console.print(
                    f"[bold red]{name} is mid-promotion ({racing[0]}); rewriting "
                    "its trace now could race the fit. Exclude it or wait.[/bold red]"
                )
                return 1

    rows = []
    for directory in directories:
        if not os.path.isfile(os.path.join(directory, TRACE_FILENAME)):
            console.print(f"[yellow]{os.path.basename(directory)}: no trace.nc[/yellow]")
            continue
        try:
            rows.append(compact_one(directory, tier, dry_run=args.dry_run))
        except Exception as exc:
            console.print(f"[bold red]{os.path.basename(directory)}: {exc}[/bold red]")
            rows.append({"model": os.path.basename(directory), "status": f"FAILED: {exc}"})

    if rows:
        import pandas as pd

        dataframe_table(
            pd.DataFrame(rows),
            title=f"Trace persistence -> {tier.value}"
            + (" [dry run]" if args.dry_run else ""),
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
