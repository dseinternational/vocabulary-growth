#!/usr/bin/env python
# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Does the typically developing tau-kappa ridge appear when replication is thinned?

Drafted by an LLM-based AI tool (Claude Code/Fable 5.1).

The question
------------
Every typically developing model with a child effect sits below the 0.3 energy
BFMI threshold and carries a strong positive posterior correlation between the
child scale and the young-age Beta-Binomial concentration -- +0.57 to +0.76 in
VG12, VG13, VG21 and VG23 (``notes/202609061900-td-bfmi-is-the-tau-kappa-ridge.md``).
The Down syndrome control, VG08, has the same two random effects on the same
probability scale and **no** ridge: corr(tau_subj_u, kappa_young_u) = -0.001.
The obvious difference is replication -- 46.6% of Down syndrome children are
seen twice or more, against 17.2% in VG12 -- and #229 proposes the decisive test:
thin the Down syndrome pool to the typically developing replication profile and
see whether the ridge appears.

That is what this fits. Three arms of VG08, on frames derived from its own
prepared analysis frame:

    baseline   the frame as fitted (1,708 rows, 943 children, 46.6% repeaters)
    thinned    the same 943 children; enough repeaters truncated to one randomly
               chosen visit that the repeater share falls to VG12's 17.2%. Only
               extra visits leave, so the number of child effects is unchanged
               and replication is the only thing that moves.
    control    the same number of rows as `thinned` removes, dropped uniformly at
               random from the full frame. Children can vanish and the repeater
               share barely moves, so this separates "fewer rows" from "less
               replication".

Reading the result (#229, 2026-09-06):

    thinned shows the ridge, control does not   replication is the mechanism
    both show it                                it is sample size, not replication
    neither shows it                            the difference between the two
                                                populations is something else

"Shows the ridge" is read on corr(tau_subj_u, kappa at the young anchor) and on
the energy BFMI beside it, against the baseline arm fitted on the same machine
at the same tier -- not against the model of record's numbers, which were
measured on the VM.

What this is not
----------------
Not a refit of any model of record and not a registered sensitivity variant: the
arms are frames, not definition fields, and adding a field to
``BivariateModelDefinition`` would invalidate every bivariate fit of record.
Like the other harnesses here it writes to its own output root and is a dated
record of how a number was obtained, not an entry point.

Usage::

    uv run python scripts/experiments/vg08_replication_thinning.py all \
        --output-dir output/experiments/vg08-replication-thinning
    uv run python scripts/experiments/vg08_replication_thinning.py thinned --config test ...

``--config`` defaults to ``rep``: BFMI and a posterior correlation are properties
of the posterior geometry, and the model of record's figures are ``rep`` ones.
Each arm writes ``<output-dir>/models/VG08-replication-<arm>/`` (trace and
diagnostics) and ``<output-dir>/<arm>.json``; ``all`` (or a later ``score``
call) also writes ``<output-dir>/summary.csv`` and ``summary.md``.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import time
from multiprocessing import freeze_support

import numpy as np
import pandas as pd

ARMS = ("baseline", "thinned", "control")
MODEL_KEY = "vg08"
NAME_PREFIX = "VG08-replication"
# VG12's repeater share on its own prepared frame (notes/202609061900, table
# "frame / rows / children / mean obs/child / children with >=2 visits").
TARGET_REPEATER_SHARE = 0.172
CHILD_KEY = "subject_key"


# ----------------------------------------------------------------------------
# Frames
# ----------------------------------------------------------------------------


def replication_profile(df: pd.DataFrame) -> dict:
    """Rows, children, repeater share and mean visits for a frame."""
    per_child = df.groupby(CHILD_KEY, sort=False).size()
    return {
        "rows": int(len(df)),
        "children": int(len(per_child)),
        "repeaters": int((per_child >= 2).sum()),
        "repeater_share": float((per_child >= 2).mean()),
        "rows_on_repeaters": int(per_child[per_child >= 2].sum()),
        "mean_obs_per_child": float(per_child.mean()),
        "understood_rows": int(df["understood"].notna().sum()),
        "spoken_rows": int(df["spoken"].notna().sum()),
    }


def thin_replication(
    df: pd.DataFrame, rng: np.random.Generator, target_share: float
) -> pd.DataFrame:
    """Truncate repeaters to one random visit until the repeater share hits the target.

    Every child stays in the frame. Repeaters to keep intact are chosen at random
    from the repeaters; each of the rest keeps one visit, chosen at random rather
    than the first, so the truncated children's ages stay spread over the range
    the way the typically developing singletons' are.
    """
    per_child = df.groupby(CHILD_KEY, sort=False).size()
    repeaters = per_child.index[per_child >= 2].to_numpy()
    n_children = len(per_child)
    n_keep = int(round(target_share * n_children))
    if n_keep >= len(repeaters):
        raise ValueError(
            f"Target share {target_share} keeps {n_keep} repeaters but only "
            f"{len(repeaters)} exist; nothing to thin."
        )
    keep_intact = set(rng.choice(repeaters, size=n_keep, replace=False).tolist())
    truncate = [c for c in repeaters if c not in keep_intact]

    keep_index = []
    for child, rows in df.groupby(CHILD_KEY, sort=False).groups.items():
        rows = np.asarray(list(rows))
        if child in truncate:
            keep_index.append(rng.choice(rows, size=1))
        else:
            keep_index.append(rows)
    keep_index = np.sort(np.concatenate(keep_index))
    return df.loc[keep_index].reset_index(drop=True)


def drop_rows_at_random(
    df: pd.DataFrame, rng: np.random.Generator, n_drop: int
) -> pd.DataFrame:
    """Drop ``n_drop`` rows uniformly at random, whatever that does to children."""
    drop = rng.choice(np.arange(len(df)), size=n_drop, replace=False)
    keep = np.setdiff1d(np.arange(len(df)), drop)
    return df.iloc[keep].reset_index(drop=True)


def recode_dense(df: pd.DataFrame) -> pd.DataFrame:
    """Re-issue ``study_code`` and ``subject_code`` densely for the rows present.

    The engine sizes the study and child blocks from the maximum code plus one
    (``observation_arrays.prepare_bivariate_observations``), so a frame that has
    lost children but kept the full frame's codes would carry prior-only child
    effects with no rows behind them. The first control run did exactly that:
    769 children, 943 child-effect slots. Codes are reassigned in order of first
    appearance so the mapping is deterministic.
    """
    out = df.copy()
    for key, code in (("study", "study_code"), (CHILD_KEY, "subject_code")):
        if int(out[code].max()) + 1 == out[key].nunique():
            continue  # already dense: keep the frame builder's own coding
        labels = pd.unique(out[key])
        out[code] = out[key].map({lab: i for i, lab in enumerate(labels)}).astype(int)
    return out


def build_arm_frames(
    base: pd.DataFrame, seed: int, target_share: float
) -> dict[str, pd.DataFrame]:
    """The three arm frames, derived deterministically from ``seed``."""
    thinned = thin_replication(base, np.random.default_rng(seed), target_share)
    n_removed = len(base) - len(thinned)
    control = drop_rows_at_random(base, np.random.default_rng(seed + 1), n_removed)
    return {
        "baseline": recode_dense(base),
        "thinned": recode_dense(thinned),
        "control": recode_dense(control),
    }


# ----------------------------------------------------------------------------
# Fitting
# ----------------------------------------------------------------------------


def fit_arm(definition, df: pd.DataFrame, config: str, output_root: str, arm: str):
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
        model_name=f"{NAME_PREFIX}-{arm}",
        config_name=definition.config_name,
        output_root_dir=output_root,
        ci_prob=0.89,
        interval_kind="eti",
    )
    os.makedirs(reporting_cfg.output_dir, exist_ok=True)
    context = ModelFitContext(
        reporting=reporting_cfg,
        sampling=sampling.get_sampling_configuration(config),
        sampling_config_name=config,
    )
    context.set_model_data(bmd, df)
    configure_bivariate_priors(context, definition)
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


def bfmi_per_chain(idata) -> np.ndarray:
    # Betancourt (2016): sum of squared successive energy differences over the
    # energy variance, per chain -- the statistic the gate thresholds at 0.3.
    e = idata.sample_stats["energy"].values
    return np.array([np.sum(np.diff(r) ** 2) / np.sum((r - r.mean()) ** 2) for r in e])


def young_anchor_kappa(idata, definition) -> tuple[np.ndarray, str]:
    """kappa_u at the young slope anchor, read off the stored query grid.

    VG08's kappa block is the ``kappa_min + exp(a + b z)`` form, which emits no
    ``kappa_young`` deterministic; the query grid is stored and contains the
    anchor age, so the same quantity is read from there.
    """
    young_age = definition.slope_anchors[0]
    ages = list(definition.ages_query)
    if "kappa_u_query" in idata.posterior and young_age in ages:
        idx = ages.index(young_age)
        return idata.posterior["kappa_u_query"].values[..., idx].reshape(-1), (
            f"kappa_u_query[{young_age} mo]"
        )
    return _flat(idata, "kappa_min_u"), "kappa_min_u (query grid absent)"


def score_arm(idata, definition, gate: dict) -> dict:
    energy = idata.sample_stats["energy"].values.reshape(-1)
    tau = _flat(idata, "tau_subj_u")
    kappa_young, kappa_label = young_anchor_kappa(idata, definition)
    kappa_min = _flat(idata, "kappa_min_u")

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
    q = np.quantile(tau, [0.055, 0.5, 0.945])
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
        "ridge_tau_kappa_young": float(np.corrcoef(tau, kappa_young)[0, 1]),
        "ridge_tau_kappa_min": float(np.corrcoef(tau, kappa_min)[0, 1]),
        "kappa_young_label": kappa_label,
        "corr_tau_energy": float(np.corrcoef(tau, energy)[0, 1]),
        "corr_kappa_young_energy": float(np.corrcoef(kappa_young, energy)[0, 1]),
        "top_energy_correlates": energy_corr[:12],
        "tau_subj_u": {
            "mean": float(tau.mean()),
            "sd": float(tau.std()),
            "cv": float(tau.std() / tau.mean()),
            "eti89": [float(q[0]), float(q[2])],
        },
        "kappa_young_u": {
            "mean": float(kappa_young.mean()),
            "sd": float(kappa_young.std()),
        },
    }


def write_summary(output_root: str) -> None:
    rows = []
    for arm in ARMS:
        path = os.path.join(output_root, f"{arm}.json")
        if not os.path.isfile(path):
            continue
        with open(path, encoding="utf-8") as fh:
            rec = json.load(fh)
        f, s = rec["frame"], rec["score"]
        rows.append(
            {
                "arm": arm,
                "rows": f["rows"],
                "children": f["children"],
                "repeater_share": round(f["repeater_share"], 3),
                "mean_obs": round(f["mean_obs_per_child"], 3),
                "min_bfmi": round(s["min_bfmi"], 3),
                "ridge_tau_kappa_young": round(s["ridge_tau_kappa_young"], 3),
                "ridge_tau_kappa_min": round(s["ridge_tau_kappa_min"], 3),
                "corr_tau_energy": round(s["corr_tau_energy"], 3),
                "corr_kappa_young_energy": round(s["corr_kappa_young_energy"], 3),
                "energy_sd_ratio": (
                    round(s["energy_sd_over_sqrt_half_d"], 2)
                    if s["energy_sd_over_sqrt_half_d"] is not None
                    else None
                ),
                "divergences": s["divergences"],
                "max_rhat": round(s["max_rhat"], 4) if s["max_rhat"] is not None else None,
                "min_ess": round(s["min_ess"]) if s["min_ess"] is not None else None,
                "tau_mean": round(s["tau_subj_u"]["mean"], 4),
                "tau_sd": round(s["tau_subj_u"]["sd"], 4),
                "tau_cv": round(s["tau_subj_u"]["cv"], 4),
                "kappa_young_mean": round(s["kappa_young_u"]["mean"], 3),
                "config": rec["config"],
                "seed": rec["seed"],
                "minutes": round(rec["elapsed_seconds"] / 60, 1),
            }
        )
    if not rows:
        return
    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(output_root, "summary.csv"), index=False)
    cols = list(table.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, r in table.iterrows():
        lines.append("| " + " | ".join("" if pd.isna(r[c]) else str(r[c]) for c in cols) + " |")
    with open(os.path.join(output_root, "summary.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print()
    print(table.to_string(index=False))


# ----------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "arm",
        nargs="+",
        choices=ARMS + ("all", "score", "frames"),
        help="Arms to fit; 'all' fits the three in order, 'score' only rewrites the "
        "summary from existing arm JSON files, 'frames' prints the frame profiles and exits.",
    )
    ap.add_argument("--config", default="rep", help="Sampling configuration (default: rep).")
    ap.add_argument("--output-dir", required=True, help="Output root for this experiment; never a publication root.")
    ap.add_argument("--seed", type=int, default=47, help="RNG seed for the thinning and the random drop (default: 47).")
    ap.add_argument(
        "--target-repeater-share",
        type=float,
        default=TARGET_REPEATER_SHARE,
        help=f"Repeater share the thinned arm is brought down to (default: {TARGET_REPEATER_SHARE}, VG12's).",
    )
    args = ap.parse_args()

    from vocab_growth import environment as env
    from vocab_growth.analysis_frames import build_analysis_frame
    from vocab_growth.models.definitions import MODEL_REGISTRY

    output_root = os.path.abspath(args.output_dir)
    os.makedirs(output_root, exist_ok=True)
    env.set_output_root(output_root)

    definition = MODEL_REGISTRY[MODEL_KEY]
    base, _side = build_analysis_frame(MODEL_KEY, definition)
    frames = build_arm_frames(base, args.seed, args.target_repeater_share)
    profiles = {arm: replication_profile(df) for arm, df in frames.items()}
    print(pd.DataFrame(profiles).T.to_string())
    with open(os.path.join(output_root, "frames.json"), "w", encoding="utf-8") as fh:
        json.dump(
            {"seed": args.seed, "target_repeater_share": args.target_repeater_share, "profiles": profiles},
            fh,
            indent=2,
        )

    requested = list(args.arm)
    if "frames" in requested:
        return
    if "all" in requested:
        requested = list(ARMS)
    to_fit = [a for a in requested if a in ARMS]

    for arm in to_fit:
        print(f"\n=== {arm}: {profiles[arm]['rows']} rows, {profiles[arm]['children']} children, "
              f"{100 * profiles[arm]['repeater_share']:.1f}% repeaters ===")
        trace, gate, trace_path, elapsed = fit_arm(
            definition, frames[arm], args.config, output_root, arm
        )
        score = score_arm(trace, definition, gate)
        record = {
            "arm": arm,
            "model": MODEL_KEY,
            "config": args.config,
            "seed": args.seed,
            "target_repeater_share": args.target_repeater_share,
            "frame": profiles[arm],
            "score": score,
            "definition": dataclasses.asdict(definition),
            "trace": trace_path,
            "elapsed_seconds": elapsed,
        }
        with open(os.path.join(output_root, f"{arm}.json"), "w", encoding="utf-8") as fh:
            json.dump(record, fh, indent=2, default=str)
        print(
            f"{arm}: min BFMI {score['min_bfmi']:.3f}  "
            f"corr(tau_subj_u, {score['kappa_young_label']}) = {score['ridge_tau_kappa_young']:+.3f}  "
            f"corr(tau, energy) = {score['corr_tau_energy']:+.3f}  "
            f"divergences {score['divergences']}  max R-hat {score['max_rhat']}  "
            f"({elapsed / 60:.1f} min)"
        )

    write_summary(output_root)


if __name__ == "__main__":
    freeze_support()
    main()
