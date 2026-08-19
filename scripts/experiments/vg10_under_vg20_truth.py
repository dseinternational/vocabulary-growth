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

Two modes
---------
**Pinned** (``--rho 0.368``) is the headline experiment: the truth is a VG20
posterior draw with ``rho_uq`` overwritten to a stated value, so every replicate
is generated at the same, named correlation and the replicates differ only in
the remaining parameters and the simulation noise. This mode simulates and fits
*both* arms itself, because the recovery harness's own runs are at its own draws.

**Harness draws** (no ``--rho``) reuses whatever ``fit_recovery.py vg20`` already
simulated and scored. Its truths span the posterior (0.311-0.478 on the current
run), which makes it the dose-response companion: if the absorption scales with
the true ``rho``, that is mechanism confirmation the pinned run cannot give.

Usage::

    python scripts/experiments/vg10_under_vg20_truth.py --rho 0.368 --config test
    python scripts/experiments/vg10_under_vg20_truth.py --config rep
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

#: ``rho_uq = 2 * rho_uq_raw - 1`` with ``rho_uq_raw ~ Beta(eta, eta)``, and
#: ``rho_uq_raw`` is the free random variable the truth draw carries. Pinning the
#: correlation therefore means pinning the raw value.
RAW_NAME = "rho_uq_raw"


def raw_for(rho: float) -> float:
    """The free parameter's value for a stated correlation."""
    return (rho + 1.0) / 2.0


def pinned_suffix(rho: float) -> str:
    return f"-rho{round(rho * 1000):03d}"

#: Parameters both models carry, split by whether a difference between the arms
#: is readable. `rho_uq` is in neither: VG10 has no such parameter, and the point
#: of the experiment is where its value goes instead.
#:
#: REPORTED are the quantities the models actually report and that recover well
#: enough for a between-arm difference to mean something. The dispersion ones are
#: stated at two reference ages precisely because that is the identified
#: parameterisation (see PRIORS.md).
REPORTED = (
    "tau_subj_u",
    "tau_subj_q",
    "kappa_young_u",
    "kappa_old_u",
    "kappa_young_s",
    "kappa_old_s",
    "tau_u",
    "tau_q",
)

#: COMPONENTS are the floor-plus-excess terms the reference-age values are built
#: from. They trade off against one another and are individually unidentified at
#: recovery tiers: on VG20's own gate-2 replicate the *correctly specified* model
#: missed `kappa_excess_old_s` by +265%, `kappa_min_s` by -60% and `a_kappa_s` by
#: +75%, while the derived `kappa_old_s` was within 1%. Differences between the
#: arms on these are noise on top of noise, so they are reported separately and
#: must not be read as the answer -- ranking everything together by magnitude
#: puts precisely the unidentified terms at the top.
COMPONENTS = (
    "kappa_min_u",
    "kappa_excess_young_u",
    "kappa_excess_old_u",
    "kappa_min_s",
    "kappa_excess_young_s",
    "kappa_excess_old_s",
    "a_kappa_u",
    "b_kappa_u",
    "a_kappa_s",
    "b_kappa_s",
)

SHARED = REPORTED + COMPONENTS


def source_definition(rho: float | None):
    """VG20, suffixed when the correlation is pinned.

    A distinct ``config_name`` gives the pinned run its own simulation directory,
    which matters while ``fit_recovery.py vg20`` may be running against the
    unsuffixed ones.
    """
    base = MODEL_REGISTRY[SOURCE_KEY]
    if rho is None:
        return base
    return dataclasses.replace(
        base, config_name=f"{base.config_name}{pinned_suffix(rho)}"
    )


def target_definition(rho: float | None = None):
    """VG10, redirected so it cannot overwrite its own recovery baseline."""
    base = MODEL_REGISTRY[TARGET_KEY]
    tail = SUFFIX if rho is None else f"{SUFFIX}{pinned_suffix(rho)}"
    return dataclasses.replace(base, config_name=f"{base.config_name}{tail}")


def simulate_pinned(replicate: int, rho: float, config: str) -> None:
    """Simulate one dataset from VG20 with ``rho_uq`` pinned to ``rho``.

    ``simulate_replicate`` reads its truth from the *passed* definition's own
    trace, which a suffixed definition does not have, and takes whatever
    ``rho_uq_raw`` the draw carries. Both are handled by wrapping
    ``truth_from_trace``: it is redirected to the real VG20 model-of-record
    trace, and the returned draw's ``rho_uq_raw`` is overwritten with the value
    corresponding to ``rho``. Everything else in the draw is left alone, so the
    replicates still span the posterior in every parameter but this one.

    Patching rather than reimplementing keeps the simulation path identical to
    the harness's -- same build, same forward draw, same provenance record.
    """
    from vocab_growth.recovery import simulate as sim

    record_trace = os.path.join(
        env.output_root(),
        "models",
        f"{MODEL_REGISTRY[SOURCE_KEY].model_id}-"
        f"{MODEL_REGISTRY[SOURCE_KEY].config_name}",
        "trace.nc",
    )
    if not os.path.isfile(record_trace):
        raise SystemExit(f"No VG20 model-of-record trace at {record_trace}.")

    raw = raw_for(rho)
    original = sim.truth_from_trace

    def patched(trace_path, free_rv_names, *, replicate):  # noqa: ARG001
        truth = original(record_trace, free_rv_names, replicate=replicate)
        posterior = truth.tree["posterior"].to_dataset()
        if RAW_NAME not in posterior.data_vars:
            raise SystemExit(f"Truth draw has no {RAW_NAME!r}.")
        was = float(np.asarray(posterior[RAW_NAME].values).ravel()[0])
        posterior[RAW_NAME] = xr.zeros_like(posterior[RAW_NAME]) + raw
        truth.tree["posterior"] = xr.DataTree(posterior)
        truth.provenance["rho_uq_pinned_to"] = rho
        truth.provenance["rho_uq_raw_was"] = was
        truth.provenance["rho_uq_was"] = 2.0 * was - 1.0
        print(
            f"    pinned rho_uq {2.0 * was - 1.0:+.4f} -> {rho:+.4f} "
            f"(raw {was:.4f} -> {raw:.4f})"
        )
        return truth

    sim.truth_from_trace = patched
    try:
        print(f"\n=== replicate {replicate:02d}: simulating from VG20 at rho={rho} ===")
        sim.simulate_replicate(
            SOURCE_KEY,
            config,
            replicate=replicate,
            truth_source="posterior",
            definition=source_definition(rho),
        )
    finally:
        sim.truth_from_trace = original


def fit_arm(replicate: int, config: str, model_key: str, definition, source_def) -> None:
    """Fit one arm to the frame ``source_def`` simulated for ``replicate``."""
    source_dir = simulation_dir(source_def, replicate, env.output_root())
    if not os.path.isdir(source_dir):
        raise SystemExit(f"No simulation at {source_dir}; simulate first.")
    frame, _truth, record = load_simulation(source_dir)

    fit_definition = make_recovery_definition(definition, replicate)
    stages = recovery_target(model_key).resolve_stages(fit_definition)
    if stages[0][0] != PREPARE_STAGE_NAME:
        raise RuntimeError(f"Unexpected first stage {stages[0][0]!r}.")
    # Same substitution the recovery harness makes: the engine's own pipeline,
    # with data preparation replaced by a loader for the synthetic frame.
    stages[0] = (
        "Load VG20-simulated data",
        _loader_stage(frame, fit_definition, record),
    )
    print(
        f"\n=== replicate {replicate:02d}: fitting "
        f"{fit_definition.model_id} to VG20's data ==="
    )
    run_fit_pipeline(config, fit_definition, stages=stages)


def _truth_values(replicate: int, source_def) -> dict[str, float]:
    path = os.path.join(
        simulation_dir(source_def, replicate, env.output_root()), "truth.nc"
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


def _control(replicate: int, rho: float | None):
    """Correctly-specified arm: this run's own VG20 fit, or the harness's scores."""
    if rho is not None:
        label = recovery_label(source_definition(rho), replicate)
        return _posterior_summary(
            os.path.join(env.output_root(), "models", label), SHARED
        )
    path = os.path.join(
        env.output_root(),
        "comparisons",
        "recovery",
        f"recovery_vg20_r{replicate:02d}.csv",
    )
    if not os.path.isfile(path):
        return None
    scores = pd.read_csv(path).set_index("quantity")
    return {
        name: (
            float(scores.loc[name, "posterior_median"]),
            float(scores.loc[name, "posterior_sd"]),
        )
        for name in SHARED
        if name in scores.index
    }


def interval_widths(replicates, rho: float | None) -> pd.DataFrame:
    """Subject-marginal interval widths, mis-specified arm against control.

    The parameter table can only show where forcing ``rho = 0`` moves an
    *estimate*. It cannot show the cost that matters, because the correlation
    enters the child-level predictive directly: the subject-marginal draw takes
    the two deviates from the joint distribution, so a model without ``rho``
    cannot express the compounding of ``p_U`` and ``q`` however well its other
    parameters are recovered.

    On the real data that gap is 9-33% (gate 3 of #224). Here the data are
    simulated at a known ``rho``, so the same comparison says whether that width
    difference is the whole of the mis-specification cost.
    """
    rows = []
    for replicate in replicates:
        arms = {
            "VG20": recovery_label(source_definition(rho), replicate),
            "VG10": recovery_label(target_definition(rho), replicate),
        }
        frames = {}
        for arm, label in arms.items():
            path = os.path.join(
                env.output_root(), "models", label, "posterior_summary_s.csv"
            )
            if not os.path.isfile(path):
                break
            frames[arm] = pd.read_csv(path)
        if len(frames) != 2:
            continue
        merged = frames["VG10"].merge(
            frames["VG20"], on="age_months", suffixes=("_t", "_c")
        )
        for _, r in merged.iterrows():
            w_t = r["Ey_subject_marginal_ci_hi_t"] - r["Ey_subject_marginal_ci_lo_t"]
            w_c = r["Ey_subject_marginal_ci_hi_c"] - r["Ey_subject_marginal_ci_lo_c"]
            if w_c <= 0:
                continue
            rows.append(
                {
                    "replicate": f"r{replicate:02d}",
                    "age_months": r["age_months"],
                    "width_VG10": w_t,
                    "width_VG20": w_c,
                    "ratio_VG10_over_VG20": w_t / w_c,
                }
            )
    return pd.DataFrame(rows)


def score(replicates, rho: float | None) -> pd.DataFrame:
    source_def = source_definition(rho)
    rows = []
    for replicate in replicates:
        truth = _truth_values(replicate, source_def)
        label = recovery_label(target_definition(rho), replicate)
        treated = _posterior_summary(
            os.path.join(env.output_root(), "models", label), SHARED
        )
        control = _control(replicate, rho)
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
            if control is not None and name in control:
                med_a, sd_a = control[name]
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
    parser.add_argument(
        "--rho",
        type=float,
        default=None,
        help="Pin rho_uq to this value and simulate both arms (e.g. 0.368).",
    )
    parser.add_argument("--score-only", action="store_true")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    env.set_output_root(args.output_dir)
    rho = args.rho

    if not args.score_only:
        for replicate in args.replicates:
            if rho is not None:
                simulate_pinned(replicate, rho, args.config)
                # Control first: if the correctly specified model cannot recover
                # its own data, the mis-specified comparison means nothing.
                fit_arm(
                    replicate, args.config, SOURCE_KEY, source_definition(rho),
                    source_definition(rho),
                )
            fit_arm(
                replicate, args.config, TARGET_KEY, target_definition(rho),
                source_definition(rho),
            )

    table = score(args.replicates, rho)
    if table.empty:
        raise SystemExit("Nothing scored.")
    out_dir = os.path.join(env.output_root(), "comparisons", "recovery")
    os.makedirs(out_dir, exist_ok=True)
    stem = "vg10_under_vg20_truth" + (pinned_suffix(rho) if rho is not None else "")
    path = os.path.join(out_dir, f"{stem}.csv")
    table.to_csv(path, index=False)

    pd.set_option("display.width", 220)
    print("\n=== VG10 fitted to VG20's data, against the same truth ===")
    print(table.round(4).to_string(index=False))
    print(f"\nwritten: {path}")

    if "VG10_minus_VG20_pct" in table.columns:
        cols = ["VG10_pct", "VG20_pct", "VG10_minus_VG20_pct"]
        for names, heading, gloss in (
            (
                REPORTED,
                "REPORTED quantities — this is the answer",
                "VG10_minus_VG20_pct: on identical data against an identical "
                "truth, where does\nforcing rho = 0 push each one? A bias common "
                "to both arms cancels.",
            ),
            (
                COMPONENTS,
                "COMPONENT terms — not the answer, kept for completeness",
                "Individually unidentified at recovery tiers: check VG20_pct, "
                "the correctly\nspecified arm, before reading any difference "
                "here. Where that column is large,\nthe difference beside it is "
                "noise on top of noise.",
            ),
        ):
            subset = table[table["parameter"].isin(names)]
            if subset.empty:
                continue
            print(f"\n=== mean over replicates: {heading} ===")
            print(
                subset.groupby("parameter")[cols]
                .mean()
                .sort_values("VG10_minus_VG20_pct", key=abs, ascending=False)
                .round(2)
                .to_string()
            )
            print(gloss)

    widths = interval_widths(args.replicates, rho)
    if not widths.empty:
        wpath = os.path.join(out_dir, f"{stem}-subject-marginal-widths.csv")
        widths.to_csv(wpath, index=False)
        print("\n=== spoken subject-marginal interval width, VG10 / VG20 ===")
        pivot = widths.pivot_table(
            index="age_months", columns="replicate", values="ratio_VG10_over_VG20"
        )
        print(pivot.round(3).to_string())
        print(
            f"\nmedian across all ages and replicates: "
            f"{widths['ratio_VG10_over_VG20'].median():.4f}"
        )
        print(
            "Below 1 means the mis-specified model reports intervals that are too "
            "narrow for a\nchild, which is the cost the parameter table cannot "
            "show. Real data gave 1/1.137 = 0.880."
        )
        print(f"written: {wpath}")


if __name__ == "__main__":
    main()
