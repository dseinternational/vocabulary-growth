# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Write a factor model's implied child-effect correlation matrix as a table.

VG22 emits ``subject_factor_corr`` — the 4x4 correlation over ``(b0u, b1u, b0q,
b1q)`` implied by its loadings — as a deterministic in the trace and nowhere
else. It is the quantity the model exists to estimate (the level-to-rate
coupling no other model carries), and its report could not show it: the
diagnostics table lists scalars only. This reads that one variable from
``trace.nc`` without loading the posterior, and writes
``subject_factor_corr.csv`` beside it for the report's rendered cell.

Usage::

    python scripts/emit_factor_correlation.py <fit-directory> [...]
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np

OUTPUT_FILENAME = "subject_factor_corr.csv"
VARIABLE = "subject_factor_corr"


def summarise(fit_dir: str) -> str | None:
    """Write the correlation table for one fit; return its path, or None if absent."""
    import h5netcdf

    trace = os.path.join(fit_dir, "trace.nc")
    if not os.path.isfile(trace):
        return None
    with h5netcdf.File(trace, "r") as handle:
        posterior = handle["posterior"]
        if VARIABLE not in posterior.variables:
            return None
        variable = posterior.variables[VARIABLE]
        values = np.asarray(variable[...], dtype=float)  # (chain, draw, 4, 4)
        dims = variable.dimensions
        labels = None
        if len(dims) == 4 and dims[2] in posterior.variables:
            raw = np.asarray(posterior.variables[dims[2]][...]).tolist()
            labels = [x.decode() if isinstance(x, bytes) else str(x) for x in raw]
    flat = values.reshape(-1, values.shape[-2], values.shape[-1])
    n = flat.shape[-1]
    if labels is None or len(labels) != n:
        labels = [f"effect_{i}" for i in range(n)]
    lo, hi = np.percentile(flat, [5.5, 94.5], axis=0)
    mean = flat.mean(axis=0)
    sd = flat.std(axis=0)
    path = os.path.join(fit_dir, OUTPUT_FILENAME)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["row", "column", "mean", "sd", "eti89_lb", "eti89_ub"])
        for i in range(n):
            for j in range(n):
                writer.writerow(
                    [labels[i], labels[j], f"{mean[i, j]:.4f}", f"{sd[i, j]:.4f}", f"{lo[i, j]:.4f}", f"{hi[i, j]:.4f}"]
                )
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fit_dir", nargs="+", help="Fitted output directory holding trace.nc")
    args = parser.parse_args(argv)
    status = 0
    for fit_dir in args.fit_dir:
        path = summarise(fit_dir)
        if path is None:
            print(f"{fit_dir}: no {VARIABLE} in the trace (not a factor model, or no trace)")
            status = 1
        else:
            print(f"{fit_dir}: wrote {os.path.basename(path)}")
    return status


if __name__ == "__main__":
    sys.exit(main())
