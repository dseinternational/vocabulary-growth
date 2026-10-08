#!/usr/bin/env python
# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Compare VG13 sampling with child scales nearly fixed at fitted values.

Drafted by an LLM-based AI tool (Claude Code/Opus 5).
Revised with assistance from OpenAI Codex/GPT-6.

``baseline`` fits the full frame. ``repeaters`` retains children with at least
two visits and rebuilds study and child codes. ``pinned`` fits the full frame
with narrow scale priors centred at the repeaters' posterior means.
``pinned-baseline`` centres them at the full-frame baseline means instead.

Compare BFMI, divergences and posterior correlations at the same tier. The two
pinned arms help distinguish sensitivity to the chosen scale values from the
restriction of scale uncertainty. Keeping the same number of free parameters
does not make their energy distributions or BFMI directly equivalent.

These exploratory arms discard uncertainty in their fitted centres. Better
sampling would not certify their intervals or establish a remedy for every TD
model. A zero-centred HalfNormal cannot express a positive modal scale, but
other prior families can; pinning is one candidate, not the only possible one.

Usage::

    uv run python scripts/experiments/vg13_repeaters_cut.py baseline repeaters --output-dir output/experiments/vg13-repeaters-cut
    uv run python scripts/experiments/vg13_repeaters_cut.py pinned --output-dir output/experiments/vg13-repeaters-cut

``pinned`` needs ``repeaters.json``; ``pinned-baseline`` needs ``baseline.json``.
``all`` runs all four arms in order. ``--config`` defaults to ``test``. Use a
separate output root; none of these arms is a fit of record.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import time
from multiprocessing import freeze_support
from typing import Any

import numpy as np
import pandas as pd
from dse_research_utils.statistics.diagnostics import (
    bfmi_per_chain as _shared_bfmi_per_chain,
)

ARMS = ("baseline", "repeaters", "pinned", "pinned-baseline")
MODEL_KEY = "vg13"
NAME_PREFIX = "VG13-repeaters-cut"
CHILD_KEY = "subject_key"
# The two (scale, dispersion-at-young-anchor) pairs whose ridge this measures.
# VG13 emits `kappa_young_u` / `kappa_young_s` as deterministics, so unlike
# VG08 no query-grid read is needed. Note the asymmetry in the engine's own
# naming: the production scale is `tau_subj_q` and its dispersion `kappa_*_s`.
RIDGE_PAIRS = (("tau_subj_u", "kappa_young_u"), ("tau_subj_q", "kappa_young_s"))
PINNED_SCALES = ("tau_subj_u", "tau_subj_q")
# Absolute sd of the near-degenerate pinned distribution. The scales are O(0.7),
# so this is about 0.14% relative -- the VG12 precedent's order (sd 0.001 on a
# share in [0, 1]) and far enough from zero that the truncation is inert.
PIN_SD = 0.001
# The sampler seed the shared configuration uses. #240 item 24 asks for
# "stability across independent fits", so the arms have to be repeatable at a
# second seed; 47 is the shared default and reproduces a run that named none.
DEFAULT_SEED = 47


# ----------------------------------------------------------------------------
# Frames
# ----------------------------------------------------------------------------


def replication_profile(df: pd.DataFrame) -> dict:
    per_child = df.groupby(CHILD_KEY, sort=False).size()
    return {
        "rows": int(len(df)),
        "children": int(len(per_child)),
        "repeaters": int((per_child >= 2).sum()),
        "repeater_share": float((per_child >= 2).mean()),
        "rows_on_repeaters": int(per_child[per_child >= 2].sum()),
        "mean_obs_per_child": float(per_child.mean()),
        "studies": int(df["study"].nunique()),
    }


def recode_dense(df: pd.DataFrame) -> pd.DataFrame:
    """Re-issue ``study_code`` / ``subject_code`` densely for the rows present.

    The engine sizes the study and child blocks from the maximum code plus one,
    so a subset frame keeping the full frame's codes would carry prior-only
    effects with no rows behind them (the lesson of the VG08 thinning control).
    Left alone when already dense, so the baseline keeps the builder's coding.
    """
    out = df.copy()
    for key, code in (("study", "study_code"), (CHILD_KEY, "subject_code")):
        if int(out[code].max()) + 1 == out[key].nunique():
            continue
        labels = sorted(out[key].unique())
        out[code] = out[key].map({lab: i for i, lab in enumerate(labels)}).astype(int)
    return out


def repeaters_only(df: pd.DataFrame) -> pd.DataFrame:
    per_child = df.groupby(CHILD_KEY, sort=False).size()
    keep = df[CHILD_KEY].map(per_child) >= 2
    return recode_dense(df.loc[keep].reset_index(drop=True))


# ----------------------------------------------------------------------------
# Pinning
# ----------------------------------------------------------------------------


@contextlib.contextmanager
def pinned_half_normals(pinned: dict[str, float]):
    """Replace named ``pm.HalfNormal`` draws with a near-degenerate scale.

    A ``HalfNormal`` has its mode at zero and cannot express "about 0.73" at any
    sigma, so the pin cannot be written as a member of the prior's own family
    the way VG12's Beta could. This intercepts the two calls by **name** and
    substitutes a ``TruncatedNormal`` at the pinned value; every other
    ``HalfNormal`` in the graph -- the study scales, the dispersion floors --
    passes through untouched.

    A patch rather than a definition field on purpose: this is a geometry test,
    and the registered graph must be the thing measured in the other two arms.
    The definition-level change is what the experiment's result decides.
    """
    import pymc as pm

    original = pm.HalfNormal

    def dispatch(name, *args, **kwargs):
        if name in pinned:
            return pm.TruncatedNormal(
                name, mu=float(pinned[name]), sigma=PIN_SD, lower=0.0
            )
        return original(name, *args, **kwargs)

    pm.HalfNormal = dispatch
    try:
        yield
    finally:
        pm.HalfNormal = original


# ----------------------------------------------------------------------------
# Fitting
# ----------------------------------------------------------------------------


def fit_arm(
    definition,
    df: pd.DataFrame,
    config: str,
    output_root: str,
    arm: str,
    pinned: dict[str, float] | None = None,
    seed: int = DEFAULT_SEED,
):
    """prepare -> priors -> build -> sample -> diagnostics, as ``fold_fits`` does it.

    Deliberately does not store the observation-sized deterministics: nothing
    here reads them, and the geometry is in the scalars and the energy.
    """
    import dse_research_utils.statistics.diagnostics as shared_diagnostics
    import dse_research_utils.statistics.models.data as model_data
    import dse_research_utils.statistics.models.reporting as reporting
    import dse_research_utils.statistics.models.sampling as sampling

    from vocab_growth.models.build_utils import require_valid_counts
    from vocab_growth.models.common import ModelFitContext, diagnostics_var_names
    from vocab_growth.models.common_bivariate import configure_bivariate_priors, sample
    from vocab_growth.models.common_bivariate_re import build_model_re

    has_u = df["understood"].notna().to_numpy()
    require_valid_counts(
        np.asarray(df.loc[has_u, "understood"], dtype=float),
        "understood",
        definition.n_trials,
    )
    bmd = model_data.BinomialModelData(
        X_obs=np.asarray(df["age"], dtype=float).reshape(-1, 1),
        y_obs=np.where(has_u, df["understood"].fillna(0).astype(int), 0).astype(int),
        n_trials=definition.n_trials,
    )
    reporting_cfg = reporting.ReportingConfiguration(
        model_name=(
            f"{NAME_PREFIX}-{arm}" if seed == DEFAULT_SEED
            else f"{NAME_PREFIX}-{arm}-seed{seed}"
        ),
        config_name=definition.config_name,
        output_root_dir=output_root,
        ci_prob=0.89,
        interval_kind="eti",
    )
    os.makedirs(reporting_cfg.output_dir, exist_ok=True)
    context = ModelFitContext(
        reporting=reporting_cfg,
        sampling=sampling.get_sampling_configuration(config, random_seed=seed),
        sampling_config_name=config,
    )
    context.set_model_data(bmd, df)
    configure_bivariate_priors(context, definition)
    with pinned_half_normals(pinned or {}):
        build_model_re(context, definition)
    started = time.time()
    sample(context)
    elapsed = time.time() - started

    _summary_names, gate_var_names = diagnostics_var_names(context.model)
    gate = shared_diagnostics.write_diagnostics_summary(
        context.trace, reporting_cfg.output_dir, var_names=gate_var_names
    )
    trace_path = os.path.join(reporting_cfg.output_dir, "trace.nc")
    context.trace.to_netcdf(trace_path)
    return context.trace, gate, trace_path, elapsed


# ----------------------------------------------------------------------------
# Scoring
# ----------------------------------------------------------------------------


def _flat(idata, name):
    return idata.posterior[name].values.reshape(-1)


def _scalar_names(idata):
    return [n for n, v in idata.posterior.data_vars.items() if v.values.ndim == 2]


def bfmi_per_chain(idata: Any) -> np.ndarray:
    """Return per-chain values; stop if energy diagnostics are unavailable."""
    if "energy" not in idata.sample_stats:
        raise KeyError("energy")
    values = _shared_bfmi_per_chain(idata)
    if values is None:
        raise ValueError("Energy diagnostics cannot be read in chain and draw order")
    return np.asarray(values, dtype=float)


def _summarise(x: np.ndarray) -> dict:
    q = np.quantile(x, [0.055, 0.5, 0.945])
    return {
        "mean": float(x.mean()),
        "sd": float(x.std()),
        "eti89": [float(q[0]), float(q[2])],
    }


def score_arm(idata, gate: dict) -> dict:
    energy = idata.sample_stats["energy"].values.reshape(-1)

    energy_corr = []
    for name in _scalar_names(idata):
        x = _flat(idata, name)
        if np.std(x) == 0:
            continue
        energy_corr.append((name, float(np.corrcoef(x, energy)[0, 1])))
    energy_corr.sort(key=lambda r: -abs(r[1]))

    bfmi = bfmi_per_chain(idata)
    # Free-parameter count, for the energy-SD reference sqrt(d/2) the diagnosis
    # note quotes. The sampler records the free RV names on the posterior.
    from vocab_growth.fit_artifacts import SAMPLED_PARAMETERS_ATTR

    free_names = set(json.loads(idata.posterior.attrs.get(SAMPLED_PARAMETERS_ATTR, "[]")))
    n_free = sum(
        int(np.prod(v.shape[2:]))
        for n, v in idata.posterior.data_vars.items()
        if n in free_names
    )
    checks = gate.get("checks") or {}

    ridges: dict[str, float] = {}
    posteriors: dict[str, dict] = {}
    for scale, kappa in RIDGE_PAIRS:
        tau = _flat(idata, scale)
        kap = _flat(idata, kappa)
        # A pinned scale has near-zero variance; a correlation against it is
        # not defined in any useful sense, so report None rather than noise.
        ridges[f"corr({scale},{kappa})"] = (
            float(np.corrcoef(tau, kap)[0, 1]) if np.std(tau) > 1e-6 else None
        )
        ridges[f"corr({scale},energy)"] = (
            float(np.corrcoef(tau, energy)[0, 1]) if np.std(tau) > 1e-6 else None
        )
        ridges[f"corr({kappa},energy)"] = float(np.corrcoef(kap, energy)[0, 1])
        posteriors[scale] = _summarise(tau)
        posteriors[kappa] = _summarise(kap)

    return {
        "min_bfmi": float(bfmi.min()),
        "mean_bfmi": float(bfmi.mean()),
        "bfmi_per_chain": [float(b) for b in bfmi],
        "energy_sd_over_sqrt_half_d": (
            float(np.std(energy) / np.sqrt(n_free / 2)) if n_free else None
        ),
        "n_free_parameters": int(n_free),
        "divergences": int(idata.sample_stats["diverging"].values.sum()),
        "max_rhat": gate.get("max_rhat"),
        "min_ess": gate.get("min_ess"),
        "gate_passed": bool(gate.get("passed")),
        "bfmi_ok": checks.get("bfmi"),
        "ridges": ridges,
        "posteriors": posteriors,
        "top_energy_correlates": energy_corr[:12],
    }


def arm_record_name(arm: str, seed: int) -> str:
    """File stem for one arm's record.

    A run at the default seed keeps the plain name, so it reproduces a run
    made before ``--seed`` existed; any other seed gets its own stem, because
    the stability check #240 item 24 asks for must not overwrite the result it
    is being compared against.
    """
    return arm if seed == DEFAULT_SEED else f"{arm}-seed{seed}"


def write_summary(output_root: str, seed: int = DEFAULT_SEED) -> None:
    rows = []
    for arm in ARMS:
        path = os.path.join(output_root, f"{arm_record_name(arm, seed)}.json")
        if not os.path.isfile(path):
            continue
        with open(path, encoding="utf-8") as handle:
            rec = json.load(handle)
        score = rec["score"]
        prof = rec["profile"]
        row = {
            "arm": arm,
            "rows": prof["rows"],
            "children": prof["children"],
            "studies": prof["studies"],
            "repeater_share": round(prof["repeater_share"], 4),
            "min_bfmi": round(score["min_bfmi"], 4),
            "energy_sd_ratio": (
                round(score["energy_sd_over_sqrt_half_d"], 3)
                if score["energy_sd_over_sqrt_half_d"]
                else None
            ),
            "divergences": score["divergences"],
            "max_rhat": (
                round(score["max_rhat"], 5) if score["max_rhat"] is not None else None
            ),
            "min_ess": (
                round(score["min_ess"], 1) if score["min_ess"] is not None else None
            ),
            "minutes": round(rec["elapsed_seconds"] / 60.0, 1),
        }
        for label, value in score["ridges"].items():
            row[label] = round(value, 4) if value is not None else None
        for name, summary in score["posteriors"].items():
            row[f"{name}_mean"] = round(summary["mean"], 4)
            row[f"{name}_sd"] = round(summary["sd"], 4)
        rows.append(row)
    if not rows:
        print("No arm results found; nothing to summarise.")
        return
    frame = pd.DataFrame(rows)
    path = os.path.join(output_root, "summary.csv")
    frame.to_csv(path, index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 60)
    print(f"\n{frame.to_string(index=False)}\n\nWrote {path}")


# ----------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("arms", nargs="+", help=f"one or more of {ARMS}, or 'all'")
    parser.add_argument("--config", default="test")
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="sampler seed; a value other than the default writes to its own "
        "arm files so a stability check cannot overwrite the first run",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join("output", "experiments", "vg13-repeaters-cut"),
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="rewrite summary.csv from the arm JSON already present",
    )
    args = parser.parse_args()

    output_root = os.path.abspath(args.output_dir)
    os.makedirs(output_root, exist_ok=True)

    if args.summary_only:
        write_summary(output_root, args.seed)
        return

    requested = list(ARMS) if "all" in args.arms else list(args.arms)
    unknown = [a for a in requested if a not in ARMS]
    if unknown:
        parser.error(f"unknown arm(s) {unknown}; choose from {ARMS}")
    # Keep the declared order regardless of how they were typed: `pinned` reads
    # `repeaters.json`, so a run naming both must fit the repeaters first.
    requested = [a for a in ARMS if a in requested]

    from vocab_growth.analysis_frames import build_analysis_frame
    from vocab_growth.models.definitions import MODEL_REGISTRY

    definition = MODEL_REGISTRY[MODEL_KEY]
    base, _side = build_analysis_frame(MODEL_KEY, definition)

    for arm in requested:
        pinned: dict[str, float] | None = None
        if arm == "baseline":
            frame = base
        elif arm == "repeaters":
            frame = repeaters_only(base)
        else:
            frame = base
            source = "baseline" if arm == "pinned-baseline" else "repeaters"
            path = os.path.join(
                output_root, f"{arm_record_name(source, args.seed)}.json"
            )
            if not os.path.isfile(path):
                parser.error(
                    f"{arm} needs {path}; run the '{source}' arm first so the "
                    "scales have a value to be pinned at."
                )
            with open(path, encoding="utf-8") as handle:
                rec = json.load(handle)
            pinned = {
                name: float(rec["score"]["posteriors"][name]["mean"])
                for name in PINNED_SCALES
            }

        profile = replication_profile(frame)
        print(f"\n=== {arm} ===")
        print(json.dumps(profile, indent=2))
        if pinned:
            print(f"pinned at {json.dumps(pinned)} (sd {PIN_SD})")

        idata, gate, trace_path, elapsed = fit_arm(
            definition, frame, args.config, output_root, arm,
            pinned=pinned, seed=args.seed,
        )
        score = score_arm(idata, gate)
        record = {
            "arm": arm,
            "seed": args.seed,
            "model": MODEL_KEY,
            "config": args.config,
            "profile": profile,
            "pinned": pinned,
            "pinned_from": (
                None if pinned is None
                else ("baseline" if arm == "pinned-baseline" else "repeaters")
            ),
            "pin_sd": PIN_SD if pinned else None,
            "score": score,
            "trace": trace_path,
            "elapsed_seconds": elapsed,
        }
        record_path = os.path.join(
            output_root, f"{arm_record_name(arm, args.seed)}.json"
        )
        with open(record_path, "w", encoding="utf-8") as handle:
            json.dump(record, handle, indent=2)
        print(json.dumps(score, indent=2)[:1800])
        print(f"{arm}: {elapsed / 60.0:.1f} min -> {trace_path}")

    write_summary(output_root, args.seed)


if __name__ == "__main__":
    freeze_support()
    main()
