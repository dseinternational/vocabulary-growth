# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Apply a trace-persistence tier to a fit already stored at ``full``.

Reuse the persistence policy in :mod:`vocab_growth.fit_artifacts` and record
omissions in the fit manifest. The replacement trace is checked before it
replaces the original.

``compact`` retains free parameters, sample statistics, log likelihood and
posterior predictive draws. It drops scaled random effects and any legacy
observation-sized posterior deterministics. Current fits omit those observation
arrays at every tier. ``minimal`` also drops observation-sized log likelihood
and posterior predictive entries.

Plot regeneration, leave-one-subject-out comparison and parameter-recovery
scoring require ``full`` and refuse compacted fits. Check downstream needs
before compacting; the recovery headline set is VG20, VG12 and VG15.

Usage::

    python scripts/compact_traces.py --dry-run
    python scripts/compact_traces.py VG03-age-spoken-td --tier compact

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
    # Rewrite smaller traces first to free space for larger replacements.
    # Each replacement must fit beside its original.
    return sorted(found, key=lambda d: os.path.getsize(os.path.join(d, TRACE_FILENAME)))


def _is_live(staging_entry: str) -> bool:
    """Return whether a staging entry may belong to a live process.

    Names end in ``-<timestamp>-<pid>-<hash>``. Stale directories can remain after a
    failed fit, so their presence alone does not prove a fit is running. Treat an
    unreadable name or failed probe as live and refuse the rewrite.

    Use ``psutil.pid_exists``. On Windows, ``os.kill(pid, 0)`` can terminate the
    process rather than check whether it exists.
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

        # Write beside the original so an incomplete write cannot replace it.
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

    # Refuse a rewrite if this model has a live or unidentifiable staging entry.
    # Stale entries and live fits of other models do not block compaction.
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
