#!/usr/bin/env python
# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Fit VG10 to data simulated from VG20 at a known, non-zero ``rho_uq``.

The question
------------
VG20 estimates a correlation between a child's comprehension deviation and their
production-ratio deviation; VG10 forces it to zero. Linearising
``log p_S = log p_U + log q``, the within-administration cross-outcome
covariance is::

    cov(log p_U, log p_S) = tau_u^2 + rho * tau_u * tau_q   [+ no noise term]

with no observation-level term, because the two Beta-Binomial draws are
independent across outcomes by construction. Setting ``rho = 0`` does not remove
that equation — it over-constrains it to ``cov = tau_u^2``, so the covariance the
data carry has to be absorbed somewhere else.

**Where?** ``notes/202608191200`` predicted ``tau_subj_u``. On the real data
VG10 gives 0.7970 against VG20's 0.7860, a shift of only 0.36 sigma, where full
absorption would need ``sqrt(tau_u^2 + rho*tau_u*tau_q)`` = 0.995. So the naive
prediction is a bound that is grossly violated, and the interesting possibility
is that the covariance lands in the **dispersion** instead — which would be a
concrete instance of between-child heterogeneity leaking into overdispersion,
the confound #229 is about. The real-data dispersion shifts point that way but
none exceeds 0.72 sigma, so they cannot settle it.

Simulated data can, because the truth is known.

The design
----------
For each replicate the data and the truth are *identical* between the two arms;
only the model differs. That is what isolates mis-specification from every other
source of error, including the ~5.7% low bias in the between-child scale that
both models show and which would otherwise confound a single-arm reading.

    truth      VG20's own posterior draw (known rho_uq, tau_subj_*, kappa_*)
    data       the frame `fit_recovery.py vg20` simulated from that draw
    arm A      VG20 refitted to it  -- correctly specified control, already
               scored by the recovery harness into recovery_vg20_rNN.csv
    arm B      VG10 fitted to it    -- mis-specified, this script

Arm A costs nothing extra: this script reads its scores rather than refitting.

Isolation
---------
VG10's own recovery directories hold the 2026-08-16 baseline that #225 cites, so
this script must not write there. The VG10 definition it fits carries a
``-under-vg20-truth`` suffix in its ``config_name``, which sends the output to
``VG10-...-under-vg20-truth-recovery-rNN/`` and leaves the baseline untouched.

Usage::

    python scripts/experiments/vg10_under_vg20_truth.py --config rep
    python scripts/experiments/vg10_under_vg20_truth.py --replicates 1 2
    python scripts/experiments/vg10_under_vg20_truth.py --score-only
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import sys

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from vocab_growth import environment as env  # noqa: E402
from vocab_growth.models.common import run_fit_pipeline  # noqa: E402
from vocab_growth.models.definitions import MODEL_REGISTRY  # noqa: E402
from vocab_growth.recovery.refit import (  # noqa: E402
    _loader_stage,
    make_recovery_definition,
)
from vocab_growth.recovery.simulate import (  # noqa: E402
    PREPARE_STAGE_NAME,
    load_simulation,
    recovery_label,
    simulation_dir,
)
from vocab_growth.recovery.spec import recovery_target  # noqa: E402

SOURCE_KEY = "vg20"
TARGET_KEY = "vg10"
SUFFIX = "-under-vg20-truth"

#: Parameters both models carry, so the truth is meaningful for each. `rho_uq`
#: is deliberately absent: VG10 has no such parameter, and the point of the
#: experiment is where its value goes instead.
SHARED = (
    "tau_subj_u",
    "tau_subj_q",
    "tau_u",
    "tau_q",
    "kappa_min_u",
    "kappa_excess_young_u",
    "kappa_excess_old_u",
    "kappa_min_s",
    "kappa_excess_young_s",
    "kappa_excess_old_s",
    "kappa_young_u",
    "kappa_old_u",
    "kappa_young_s",
    "kappa_old_s",
    "a_kappa_u",
    "b_kappa_u",
    "a_kappa_s",
    "b_kappa_s",
)


def target_definition():
    """VG10, redirected so it cannot overwrite its own recovery baseline."""
    base = MODEL_REGISTRY[TARGET_KEY]
    return dataclasses.replace(base, config_name=f"{base.config_name}{SUFFIX}")


def fit_one(replicate: int, config: str) -> None:
    """Fit VG10 to the frame VG20 simulated for ``replicate``."""
    root = env.output_root()
    source_dir = simulation_dir(MODEL_REGISTRY[SOURCE_KEY], replicate, root)
    if not os.path.isdir(source_dir):
        raise SystemExit(
            f"No VG20 simulation for replicate {replicate} at {source_dir}. "
            f"Run `fit_recovery.py vg20 --config {config}` first."
        )
    frame, _truth, record = load_simulation(source_dir)

    definition = make_recovery_definition(target_definition(), replicate)
    target = recovery_target(TARGET_KEY)
    stages = target.resolve_stages(definition)
    if stages[0][0] != PREPARE_STAGE_NAME:
        raise RuntimeError(f"Unexpected first stage {stages[0][0]!r}.")
    # Same substitution the recovery harness makes: the engine's own pipeline,
    # with data preparation replaced by a loader for the synthetic frame. The
    # frame came from VG20, which is the whole point.
    stages[0] = ("Load VG20-simulated data", _loader_stage(frame, definition, record))
    print(f"\n=== replicate {replicate:02d}: fitting VG10 to VG20's data ===")
    run_fit_pipeline(config, definition, stages=stages)


def _truth_values(replicate: int) -> dict[str, float]:
    root = env.output_root()
    path = os.path.join(
        simulation_dir(MODEL_REGISTRY[SOURCE_KEY], replicate, root), "truth.nc"
    )
    with xr.open_datatree(path) as tree:
        posterior = tree["posterior"].to_dataset()
        out = {}
        for name in (*SHARED, "rho_uq"):
            if name in posterior.data_vars:
                values = np.asarray(posterior[name].values, dtype=float).ravel()
                if values.size == 1:
                    out[name] = float(values[0])
    return out


def _posterior_summary(directory: str, names) -> dict[str, tuple[float, float]]:
    path = os.path.join(directory, "trace.nc")
    if not os.path.isfile(path):
        raise SystemExit(f"No trace at {path}; fit the replicate first.")
    out: dict[str, tuple[float, float]] = {}
    with xr.open_datatree(path) as tree:
        posterior = tree["posterior"].to_dataset()
        for name in names:
            if name not in posterior.data_vars:
                continue
            draws = np.asarray(posterior[name].values, dtype=float).ravel()
            out[name] = (float(np.median(draws)), float(np.std(draws, ddof=1)))
    return out


def _control_scores(replicate: int) -> pd.DataFrame | None:
    """VG20's own scores for this replicate, written by the recovery harness."""
    path = os.path.join(
        env.output_root(),
        "comparisons",
        "recovery",
        f"recovery_vg20_r{replicate:02d}.csv",
    )
    if not os.path.isfile(path):
        return None
    return pd.read_csv(path).set_index("quantity")


def score(replicates) -> pd.DataFrame:
    rows = []
    for replicate in replicates:
        truth = _truth_values(replicate)
        label = recovery_label(target_definition(), replicate)
        treated = _posterior_summary(
            os.path.join(env.output_root(), "models", label), SHARED
        )
        control = _control_scores(replicate)
        for name in SHARED:
            if name not in truth or name not in treated:
                continue
            t = truth[name]
            med_b, sd_b = treated[name]
            row = {
                "replicate": f"r{replicate:02d}",
                "rho_true": truth.get("rho_uq", np.nan),
                "parameter": name,
                "truth": t,
                "VG10_median": med_b,
                "VG10_z": (med_b - t) / sd_b if sd_b > 0 else np.nan,
                "VG10_pct": 100.0 * (med_b - t) / t if t != 0 else np.nan,
            }
            if control is not None and name in control.index:
                med_a = float(control.loc[name, "posterior_median"])
                sd_a = float(control.loc[name, "posterior_sd"])
                row["VG20_median"] = med_a
                row["VG20_z"] = (med_a - t) / sd_a if sd_a > 0 else np.nan
                row["VG20_pct"] = 100.0 * (med_a - t) / t if t != 0 else np.nan
                # The quantity the experiment is about: how far the
                # mis-specified fit sits from the correctly specified one, on
                # the same data against the same truth.
                row["VG10_minus_VG20_pct"] = row["VG10_pct"] - row["VG20_pct"]
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="rep")
    parser.add_argument("--replicates", type=int, nargs="*", default=[1, 2, 3])
    parser.add_argument("--score-only", action="store_true")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    env.set_output_root(args.output_dir)

    if not args.score_only:
        for replicate in args.replicates:
            fit_one(replicate, args.config)

    table = score(args.replicates)
    if table.empty:
        raise SystemExit("Nothing scored.")
    out_dir = os.path.join(env.output_root(), "comparisons", "recovery")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "vg10_under_vg20_truth.csv")
    table.to_csv(path, index=False)

    pd.set_option("display.width", 220)
    print("\n=== VG10 fitted to VG20's data, against the same truth ===")
    print(table.round(4).to_string(index=False))
    print(f"\nwritten: {path}")

    if "VG10_minus_VG20_pct" in table.columns:
        print("\n=== mean over replicates: mis-specification effect ===")
        summary = (
            table.groupby("parameter")[["VG10_pct", "VG20_pct", "VG10_minus_VG20_pct"]]
            .mean()
            .sort_values("VG10_minus_VG20_pct", key=abs, ascending=False)
        )
        print(summary.round(2).to_string())
        print(
            "\nVG10_minus_VG20_pct is the answer: on identical data against an "
            "identical truth,\nwhere does forcing rho = 0 push each parameter? "
            "A shared bias cancels."
        )


if __name__ == "__main__":
    main()
