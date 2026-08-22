#!/usr/bin/env python
# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Is `max R-hat <= 1.01 over every sampled parameter` measuring convergence, or size?

The convergence gate fails closed when **any** sampled parameter exceeds R-hat
1.01. The threshold itself is the current standard (Vehtari et al. 2021, which
replaced the far laxer 1.1 convention). What this script tests is the *rule
built on it*, and three specific worries about applying a fixed threshold to a
maximum over every parameter:

1. **Multiplicity.** R-hat is estimated with Monte Carlo error, so the maximum
   over N parameters is an extreme order statistic whose distribution moves up
   with N. A model with 14,600 parameters is then held to a stricter standard
   than one with 30, for reasons unrelated to whether either converged.

2. **Inconsistency with the project's own ESS floor.** R-hat's sampling error
   shrinks with effective sample size, so a fixed R-hat threshold implies a
   *minimum ESS* — and if that implied minimum is far above the explicit
   `ESS >= 400` gate, the two halves of the same gate disagree about how much
   sampling is enough.

3. **Reparameterisation.** R-hat on internal parameters is not invariant: the
   same posterior written centred or non-centred has different parameters with
   different R-hat. The reported estimands — the trajectory and query-age grids
   — are invariant. Gating on parameters therefore privileges a coordinate
   system that is an implementation choice.

Outputs, per fitted model and pooled, to `output/comparisons/rhat_calibration/`:

* `rhat_by_model.csv` — parameters screened, max R-hat, count over 1.01, and the
  same for the *reported* grids.
* `rhat_by_ess.csv` — exceedance rate binned by ESS, pooled over models. This is
  the test of worry 2.

Usage::

    python scripts/experiments/rhat_gate_calibration.py
    python scripts/experiments/rhat_gate_calibration.py --models vg11 vg12
"""

from __future__ import annotations

import argparse
import os

import arviz as az
import numpy as np
import pandas as pd
import xarray as xr

from vocab_growth import environment as env

#: Dimensions that mark a variable as a function of a *grid* rather than a
#: parameter: observations, the concatenated predictor grid, and the plot/query
#: grids the report reads.
GRID_DIMS = ("obs_id", "all_id", "plot_id", "query_id")
REPORTED_SUFFIXES = ("_plot", "_query")

#: ESS bins for the exceedance-rate table. The project's explicit floor is 400.
ESS_BINS = [0, 400, 800, 1600, 3200, 6400, 12800, np.inf]


def _is_grid(dims: tuple[str, ...]) -> bool:
    return any(d in GRID_DIMS or d.startswith("obs_") for d in dims)


def _screened_variables(post: xr.Dataset) -> list[str]:
    """Variables standing in for the gate's screened set.

    Non-grid variables, minus any scaled random effect whose ``_raw`` counterpart
    is also stored — the sampled half is the ``_raw`` one, and counting both
    would double-count a single degree of freedom. (Compacted traces have
    already dropped the scaled copies, so this only bites on ``full`` traces.)
    """
    names = []
    for name, var in post.data_vars.items():
        dims = tuple(d for d in var.dims if d not in ("chain", "draw"))
        if _is_grid(dims):
            continue
        if f"{name}_raw" in post.data_vars:
            continue
        names.append(name)
    return sorted(names)


def _reported_variables(post: xr.Dataset) -> list[str]:
    return sorted(n for n in post.data_vars if n.endswith(REPORTED_SUFFIXES))


def _rhat_ess(post: xr.Dataset, name: str) -> tuple[np.ndarray, np.ndarray]:
    d = post[name].load()
    ds = d.to_dataset(name="x")
    r = np.atleast_1d(np.asarray(az.rhat(ds)["x"].values)).ravel()
    e = np.atleast_1d(np.asarray(az.ess(ds)["x"].values)).ravel()
    del d, ds
    return r, e


def scan_model(label: str, trace_path: str) -> tuple[dict, pd.DataFrame]:
    """Return the per-model row and the per-parameter (r_hat, ess) frame."""
    t = xr.open_datatree(trace_path)
    post = t["posterior"].to_dataset()
    rows = []
    for name in _screened_variables(post):
        r, e = _rhat_ess(post, name)
        rows.append(pd.DataFrame({"model": label, "variable": name, "r_hat": r, "ess": e}))
    params = (
        pd.concat(rows, ignore_index=True)
        if rows
        else pd.DataFrame(columns=["model", "variable", "r_hat", "ess"])
    )

    rep_r, rep_e = [], []
    for name in _reported_variables(post):
        r, e = _rhat_ess(post, name)
        rep_r.append(r)
        rep_e.append(e)
    rep_r = np.concatenate(rep_r) if rep_r else np.array([np.nan])
    rep_e = np.concatenate(rep_e) if rep_e else np.array([np.nan])
    t.close()

    pr = params["r_hat"].to_numpy()
    pr = pr[~np.isnan(pr)]
    summary = {
        "model": label,
        "n_parameters": int(pr.size),
        "max_rhat_parameters": float(np.nanmax(pr)) if pr.size else np.nan,
        "n_over_1.01_parameters": int((pr > 1.01).sum()),
        "median_rhat_parameters": float(np.nanmedian(pr)) if pr.size else np.nan,
        "min_ess_parameters": float(np.nanmin(params["ess"])) if pr.size else np.nan,
        "n_reported_points": int(rep_r[~np.isnan(rep_r)].size),
        "max_rhat_reported": float(np.nanmax(rep_r)),
        "n_over_1.01_reported": int((rep_r > 1.01).sum()),
        "min_ess_reported": float(np.nanmin(rep_e)),
    }
    return summary, params


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="*", default=None, help="Output dir names or keys.")
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()
    env.set_output_root(args.output_dir)
    root = env.output_root()

    candidates: list[tuple[str, str]] = []
    models_dir = os.path.join(root, "models")
    for name in sorted(os.listdir(models_dir)):
        path = os.path.join(models_dir, name, "trace.nc")
        if os.path.isfile(path):
            candidates.append((name, path))
    # The retained VG11 fit is the case that prompted this; include it explicitly.
    failed_dir = os.path.join(root, "failed")
    if os.path.isdir(failed_dir):
        for name in sorted(os.listdir(failed_dir)):
            path = os.path.join(failed_dir, name, "trace.nc")
            if os.path.isfile(path):
                candidates.append((f"{name} [failed]", path))
    if args.models:
        wanted = [m.lower() for m in args.models]
        candidates = [c for c in candidates if any(w in c[0].lower() for w in wanted)]

    out_dir = os.path.join(root, "comparisons", "rhat_calibration")
    os.makedirs(out_dir, exist_ok=True)

    # Write per-model rows as they are produced, and skip any already present.
    # A scan of the whole family takes tens of minutes across multi-gigabyte
    # traces; losing all of it to an interruption once was enough.
    partial_path = os.path.join(out_dir, "rhat_by_model_partial.csv")
    params_path = os.path.join(out_dir, "rhat_by_parameter.csv")
    done: set[str] = set()
    if os.path.isfile(partial_path):
        done = set(pd.read_csv(partial_path)["model"])
        print(f"[resume] {len(done)} model(s) already scanned; skipping them")

    summaries, frames = [], []
    for label, path in candidates:
        if label in done:
            continue
        size = os.path.getsize(path) / 1024**3
        print(f"[scan] {label}  ({size:.1f} GiB)", flush=True)
        try:
            summary, params = scan_model(label, path)
        except Exception as exc:  # noqa: BLE001 - one bad trace must not sink the scan
            print(f"       failed: {type(exc).__name__}: {exc}", flush=True)
            continue
        summaries.append(summary)
        frames.append(params)
        pd.DataFrame([summary]).to_csv(
            partial_path, mode="a", header=not os.path.isfile(partial_path), index=False
        )
        params.to_csv(
            params_path, mode="a", header=not os.path.isfile(params_path), index=False
        )
        print(
            f"       {summary['n_parameters']:6d} params  max r_hat "
            f"{summary['max_rhat_parameters']:.4f}  "
            f"#>1.01 {summary['n_over_1.01_parameters']}   |   reported: max "
            f"{summary['max_rhat_reported']:.4f}  "
            f"#>1.01 {summary['n_over_1.01_reported']}",
            flush=True,
        )

    if not (summaries or done):
        print("nothing scanned")
        return 1

    # Aggregate from the durable per-model file so a resumed scan sees every row.
    by_model = pd.read_csv(partial_path)
    allp_src = pd.read_csv(params_path)
    by_model.to_csv(os.path.join(out_dir, "rhat_by_model.csv"), index=False)

    allp = allp_src.dropna(subset=["r_hat", "ess"])
    allp["ess_bin"] = pd.cut(allp["ess"], ESS_BINS)
    by_ess = (
        allp.groupby("ess_bin", observed=True)
        .agg(
            n=("r_hat", "size"),
            median_rhat=("r_hat", "median"),
            p99_rhat=("r_hat", lambda s: float(np.percentile(s, 99))),
            max_rhat=("r_hat", "max"),
            frac_over_1_01=("r_hat", lambda s: float((s > 1.01).mean())),
        )
        .reset_index()
    )
    by_ess.to_csv(os.path.join(out_dir, "rhat_by_ess.csv"), index=False)

    print("\n=== per model ===")
    print(by_model.to_string(index=False))
    print("\n=== exceedance by ESS (pooled over models) ===")
    print(by_ess.to_string(index=False))
    print(f"\nwritten to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
