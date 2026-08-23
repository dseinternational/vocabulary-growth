# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Score the singleton-marginalisation arms against each other.

Reads the arms `marginal_arm.py` writes into a throwaway output root and prints
the three things the bench has to answer
(notes/202608231745-singleton-marginalisation.md §7):

1. **equivalence** -- the marginalisation is exact, so `tau_subject`, `kappa` and
   the reported trajectory must agree with the explicit arm within Monte Carlo
   error. The printed z is the difference in posterior means over the combined
   Monte Carlo standard error, so |z| above about 3 is a bug rather than noise.
2. **geometry** -- energy BFMI, divergences, max R-hat, minimum ESS, and the
   `tau_subject`/`kappa_young` ridge correlation, comparable with the four arms
   of notes/202608050900-td-hierarchical-geometry.md §9.
3. **cost** -- effective samples per **gradient evaluation**, which is the
   comparison that survives running the arms on different machines or at
   different times: a marginalised gradient costs about 21x an explicit one on
   VG12, so the lever pays only if it buys that back in leapfrog steps or in
   effective samples per step. Wall-clock is printed beside it, from the fit
   state, and covers the whole pipeline rather than sampling alone.

Usage: compare_marginal_arms.py --output-dir DIR [--model VG12]
                                [--gradient-cost 23.0]
"""

import argparse
import json
import os
import re

import arviz as az
import numpy as np
import xarray as xr

EQUIVALENCE_PARAMETERS = (
    "tau_subject",
    "kappa_young",
    "kappa_old",
    "v_total",
    "subject_variance_share",
    "tau",
    "eta",
    "ell",
)


def _open(path, group):
    try:
        return xr.open_dataset(path, group=group)
    except (OSError, KeyError):
        return None


def _bfmi(sample_stats):
    energy = sample_stats["energy"].values
    return np.array(
        [
            np.sum(np.diff(row) ** 2) / np.sum((row - row.mean()) ** 2)
            for row in energy
        ]
    )


def _scalar(posterior, name):
    if name not in posterior.data_vars:
        return None
    values = posterior[name].values
    return values.reshape(-1) if values.ndim == 2 else None


def _wall_clock_seconds(directory):
    path = os.path.join(directory, "fit_state.json")
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as source:
        state = json.load(source)
    started, completed = state.get("started_at_utc"), state.get("completed_at_utc")
    if not (started and completed):
        return None
    from datetime import datetime

    return (
        datetime.fromisoformat(completed) - datetime.fromisoformat(started)
    ).total_seconds()


def _arm_summary(directory):
    trace = os.path.join(directory, "trace.nc")
    posterior = _open(trace, "posterior")
    sample_stats = _open(trace, "sample_stats")
    if posterior is None or sample_stats is None:
        return None

    free = [
        name
        for name, variable in posterior.data_vars.items()
        if variable.dims[:2] == ("chain", "draw")
    ]
    ess = az.ess(posterior[free])
    ess_values = np.hstack([ess[name].values.ravel() for name in ess.data_vars])
    rhat = az.rhat(posterior[free])
    rhat_values = np.hstack([rhat[name].values.ravel() for name in rhat.data_vars])

    divergences = (
        int(sample_stats["diverging"].values.sum())
        if "diverging" in sample_stats
        else None
    )
    gradients = (
        int(sample_stats["n_steps"].values.sum())
        if "n_steps" in sample_stats
        else None
    )
    tau_subject = _scalar(posterior, "tau_subject")
    kappa_young = _scalar(posterior, "kappa_young")
    ridge = (
        float(np.corrcoef(tau_subject, kappa_young)[0, 1])
        if tau_subject is not None and kappa_young is not None
        else float("nan")
    )
    return {
        "directory": directory,
        "posterior": posterior,
        "draws": int(posterior.sizes["chain"] * posterior.sizes["draw"]),
        "dimensions": sum(
            int(np.prod(posterior[name].shape[2:])) for name in free
        ),
        "min_bfmi": float(_bfmi(sample_stats).min()),
        "divergences": divergences,
        "max_rhat": float(np.nanmax(rhat_values)),
        "min_ess": float(np.nanmin(ess_values)),
        "gradients": gradients,
        "ridge": ridge,
        "seconds": _wall_clock_seconds(directory),
    }


def _equivalence(reference, arm):
    rows = []
    for name in EQUIVALENCE_PARAMETERS:
        left = _scalar(reference["posterior"], name)
        right = _scalar(arm["posterior"], name)
        if left is None or right is None:
            continue
        left_ess = float(az.ess(reference["posterior"][name]).values)
        right_ess = float(az.ess(arm["posterior"][name]).values)
        mcse = np.sqrt(left.var() / max(left_ess, 1.0) + right.var() / max(right_ess, 1.0))
        rows.append(
            (
                name,
                float(left.mean()),
                float(right.mean()),
                float((right.mean() - left.mean()) / mcse) if mcse > 0 else np.nan,
            )
        )
    return rows


parser = argparse.ArgumentParser()
parser.add_argument("--output-dir", required=True)
parser.add_argument("--model", default="VG12")
parser.add_argument(
    "--gradient-cost",
    type=float,
    default=21.0,
    help="cost of a marginalised gradient relative to an explicit one, for the "
    "effective-samples-per-unit-work column (VG12 at full size: 21)",
)
args = parser.parse_args()

root = os.path.join(args.output_dir, "models")
pattern = re.compile(rf"^{re.escape(args.model)}-marg-", re.IGNORECASE)
directories = sorted(
    os.path.join(root, name) for name in os.listdir(root) if pattern.match(name)
)
if not directories:
    raise SystemExit(f"No {args.model} marginalisation arms under {root}.")

arms = {}
for directory in directories:
    summary = _arm_summary(directory)
    label = os.path.basename(directory)
    if summary is None:
        print(f"  {label}: no readable trace (still running, or it failed)")
        continue
    arms[label] = summary

print(f"\n{'arm':<34} {'dims':>7} {'BFMI':>7} {'div':>5} {'R-hat':>7} {'minESS':>8} {'ridge':>7}")
for label, arm in arms.items():
    print(
        f"{label:<34} {arm['dimensions']:7d} {arm['min_bfmi']:7.3f} "
        f"{arm['divergences'] if arm['divergences'] is not None else -1:5d} "
        f"{arm['max_rhat']:7.4f} {arm['min_ess']:8.0f} {arm['ridge']:7.3f}"
    )

print(f"\n{'arm':<34} {'gradients':>12} {'ESS/1e6 grad':>13} {'wall clock':>12}")
for label, arm in arms.items():
    is_marginal = "explicit" not in label
    weight = args.gradient_cost if is_marginal else 1.0
    work = arm["gradients"] * weight if arm["gradients"] else None
    per_work = arm["min_ess"] / work * 1e6 if work else float("nan")
    clock = f"{arm['seconds'] / 60:.1f} min" if arm["seconds"] else "-"
    print(
        f"{label:<34} {arm['gradients'] if arm['gradients'] else -1:12d} "
        f"{per_work:13.2f} {clock:>12}"
    )
print(
    f"  (ESS per million gradient evaluations, with a marginalised gradient "
    f"weighted {args.gradient_cost}x; wall clock covers the whole pipeline.)"
)

reference_label = next((label for label in arms if "explicit" in label), None)
if reference_label is None:
    raise SystemExit("\nNo explicit arm to compare against; equivalence not checked.")
for label, arm in arms.items():
    if label == reference_label:
        continue
    print(f"\nequivalence: {label} against {reference_label}")
    print(f"  {'parameter':<24} {'explicit':>12} {'marginal':>12} {'z':>7}")
    for name, left, right, z in _equivalence(arms[reference_label], arm):
        print(f"  {name:<24} {left:12.4f} {right:12.4f} {z:7.2f}")
