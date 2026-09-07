#!/usr/bin/env python
# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Does a repeaters-calibrated share prior repair VG12's tau-kappa geometry?

Drafted by an LLM-based AI tool (Claude Code/Fable 5.1).

The question
------------
``notes/202609072200-vg08-replication-thinning.md`` showed that the typically
developing models' low energy BFMI and their tau-kappa ridge are what a child
effect looks like when most children are observed once: thinning VG08's
replication to VG12's profile reproduces both on Down syndrome data. No
coordinate change supplies the information a second visit carries, so the
remedy has to supply information. The cheapest candidate (#229 option 2,
generalised) is to let the children who *can* identify the split -- VG12's
1,000 children with two or more visits -- set the prior on
``subject_variance_share`` for the full fit, so the 4,800 singletons contribute
to the trajectory and the dispersion under a split the repeaters have already
determined, rather than prising it apart by functional form.

This is the pre-refit test of that idea, at ``test`` tier, on VG12 -- the model
with the worst BFMI (0.208) and the strongest ridge (+0.76), where a fix that
does nothing here does nothing anywhere. Arms:

    baseline     VG12 as registered, on its full frame
    repeaters    VG12 as registered, on the children with >= 2 visits only
    calibrated   VG12 on its full frame, with the share prior's Beta moment-
                 matched to the repeaters arm's posterior on
                 ``subject_variance_share``
    calibrated-half  as `calibrated` with the Beta concentration halved (same
                 mean, wider), to see how much of the effect is the prior's
                 tightness rather than its location
    calibrated-fixed  as `calibrated` with the concentration multiplied by a
                 thousand (sd about 0.001): the split is effectively pinned at
                 the repeaters' value, a cut rather than a prior. Added after
                 the first three arms ran: the full frame's own posterior on the
                 share (sd 0.028) is *tighter* than the repeaters' (sd 0.035),
                 so a prior at the repeaters' strength cannot dominate the split
                 the singletons impose through the likelihood's shape, and the
                 question becomes whether pinning it repairs the geometry at all

Read: min BFMI, corr(tau_subject, kappa_young), the energy correlates and the
divergence count, each arm against the baseline fitted at the same tier on the
same machine. If the calibrated arm clears 0.3 and the ridge falls, option 2 is
a candidate graph change for the TD refit. If it does not, the prior is not the
lever.

What this is not
----------------
Not a fit of record and not a registered variant. The calibrated arm uses the
repeaters' posterior as a prior and then fits the repeaters again inside the
full frame, so their information on the split is counted twice; that is a
geometry test, not the final design, and the note says so. Writes to its own
output root, never a publication root.

Usage::

    uv run python scripts/experiments/vg12_repeaters_prior.py baseline repeaters --output-dir output/experiments/vg12-repeaters-prior
    uv run python scripts/experiments/vg12_repeaters_prior.py calibrated calibrated-half --output-dir output/experiments/vg12-repeaters-prior

The calibrated arms need ``repeaters.json`` in the output root; ``all`` runs the
four in order. ``--config`` defaults to ``test``.
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

ARMS = ("baseline", "repeaters", "calibrated", "calibrated-half", "calibrated-fixed")
# Beta concentration multipliers for the tempered and the near-fixed arms.
CONCENTRATION_SCALE = {"calibrated": 1.0, "calibrated-half": 0.5, "calibrated-fixed": 1000.0}
MODEL_KEY = "vg12"
NAME_PREFIX = "VG12-repeaters-prior"
CHILD_KEY = "subject_key"


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
# Calibration
# ----------------------------------------------------------------------------


def beta_from_moments(mean: float, sd: float) -> tuple[float, float]:
    """Beta(alpha, beta) with the given mean and standard deviation."""
    var = sd**2
    if not 0.0 < mean < 1.0 or var <= 0.0 or var >= mean * (1.0 - mean):
        raise ValueError(f"No Beta has mean {mean} and sd {sd}.")
    concentration = mean * (1.0 - mean) / var - 1.0
    return mean * concentration, (1.0 - mean) * concentration


def calibrated_definition(base, share_mean: float, share_sd: float, *, concentration_scale: float = 1.0):
    """``base`` with the share prior replaced by a moment-matched Beta.

    ``concentration_scale`` < 1 widens the Beta about the same mean, for the
    tempered arm. Only the share prior moves: the total-scatter prior and
    ``reference_proportion`` stay as registered, because the budget is what the
    full frame identifies well already (#225: ``v_total`` recovers cleanly).
    """
    alpha, beta = beta_from_moments(share_mean, share_sd)
    alpha *= concentration_scale
    beta *= concentration_scale
    vp = dataclasses.replace(
        base.subject_variance_partition, share_alpha=float(alpha), share_beta=float(beta)
    )
    return dataclasses.replace(base, subject_variance_partition=vp), (float(alpha), float(beta))


# ----------------------------------------------------------------------------
# Fitting
# ----------------------------------------------------------------------------


def fit_arm(definition, df: pd.DataFrame, config: str, output_root: str, arm: str):
    """prepare -> priors -> build -> sample -> diagnostics on a given frame.

    The engine's own stages, minus data loading (the frame is supplied), the
    prior predictive figures and everything after diagnostics. The
    observation-level deterministics are not stored.
    """
    import dse_research_utils.statistics.diagnostics as shared_diagnostics
    import dse_research_utils.statistics.models.data as model_data
    import dse_research_utils.statistics.models.reporting as reporting
    import dse_research_utils.statistics.models.sampling as sampling

    from vocab_growth.models.build_utils import require_valid_counts
    from vocab_growth.models.common import (
        ModelFitContext,
        configure_univariate_priors,
        diagnostics_var_names,
        sample,
    )
    from vocab_growth.models.common_univariate_re import build_univariate_re_model

    y_col = definition.outcome.value
    y_values = np.asarray(df[y_col], dtype=float)
    require_valid_counts(y_values, y_col, definition.n_trials)
    bmd = model_data.BinomialModelData(
        X_obs=np.asarray(df["age"], dtype=float).reshape(-1, 1),
        y_obs=y_values.astype(int),
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
    configure_univariate_priors(context, definition)
    build_univariate_re_model(context, definition)
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
    e = idata.sample_stats["energy"].values
    return np.array([np.sum(np.diff(r) ** 2) / np.sum((r - r.mean()) ** 2) for r in e])


def _summary(x: np.ndarray) -> dict:
    q = np.quantile(x, [0.055, 0.5, 0.945])
    return {
        "mean": float(x.mean()),
        "sd": float(x.std()),
        "median": float(q[1]),
        "eti89": [float(q[0]), float(q[2])],
    }


def score_arm(idata, gate: dict) -> dict:
    from vocab_growth.fit_artifacts import SAMPLED_PARAMETERS_ATTR

    energy = idata.sample_stats["energy"].values.reshape(-1)
    tau = _flat(idata, "tau_subject")
    kappa_young = _flat(idata, "kappa_young")

    energy_corr = []
    for name in _scalar_names(idata):
        x = _flat(idata, name)
        if np.std(x) == 0:
            continue
        energy_corr.append((name, float(np.corrcoef(x, energy)[0, 1])))
    energy_corr.sort(key=lambda r: -abs(r[1]))

    free_names = set(json.loads(idata.posterior.attrs.get(SAMPLED_PARAMETERS_ATTR, "[]")))
    n_free = sum(
        int(np.prod(v.shape[2:])) for n, v in idata.posterior.data_vars.items() if n in free_names
    )
    bfmi = bfmi_per_chain(idata)
    checks = gate.get("checks") or {}
    out = {
        "min_bfmi": float(bfmi.min()),
        "mean_bfmi": float(bfmi.mean()),
        "bfmi_per_chain": [float(b) for b in bfmi],
        "energy_sd_over_sqrt_half_d": float(np.std(energy) / np.sqrt(n_free / 2)) if n_free else None,
        "n_free_parameters": int(n_free),
        "divergences": int(idata.sample_stats["diverging"].values.sum()),
        "max_rhat": gate.get("max_rhat"),
        "min_ess": gate.get("min_ess"),
        "gate_passed": bool(gate.get("passed")),
        "bfmi_ok": checks.get("bfmi"),
        "ridge_tau_kappa_young": float(np.corrcoef(tau, kappa_young)[0, 1]),
        "corr_tau_energy": float(np.corrcoef(tau, energy)[0, 1]),
        "corr_kappa_young_energy": float(np.corrcoef(kappa_young, energy)[0, 1]),
        "top_energy_correlates": energy_corr[:12],
        "tau_subject": _summary(tau),
        "kappa_young": _summary(kappa_young),
    }
    for name in ("subject_variance_share", "v_total", "kappa_min", "kappa_excess_young"):
        if name in idata.posterior:
            out[name] = _summary(_flat(idata, name))
    return out


def write_summary(output_root: str) -> None:
    rows = []
    for arm in ARMS:
        path = os.path.join(output_root, f"{arm}.json")
        if not os.path.isfile(path):
            continue
        with open(path, encoding="utf-8") as fh:
            rec = json.load(fh)
        f, s = rec["frame"], rec["score"]
        share = s.get("subject_variance_share", {})
        rows.append(
            {
                "arm": arm,
                "rows": f["rows"],
                "children": f["children"],
                "repeater_share": round(f["repeater_share"], 3),
                "share_prior": "Beta({:.3f}, {:.3f})".format(*rec["share_prior"]),
                "min_bfmi": round(s["min_bfmi"], 3),
                "ridge_tau_kappa_young": round(s["ridge_tau_kappa_young"], 3),
                "corr_tau_energy": round(s["corr_tau_energy"], 3),
                "corr_kappa_young_energy": round(s["corr_kappa_young_energy"], 3),
                "energy_sd_ratio": round(s["energy_sd_over_sqrt_half_d"], 2) if s["energy_sd_over_sqrt_half_d"] else None,
                "divergences": s["divergences"],
                "max_rhat": round(s["max_rhat"], 4) if s["max_rhat"] is not None else None,
                "min_ess": round(s["min_ess"]) if s["min_ess"] is not None else None,
                "tau_mean": round(s["tau_subject"]["mean"], 4),
                "tau_sd": round(s["tau_subject"]["sd"], 4),
                "share_mean": round(share["mean"], 4) if share else None,
                "share_sd": round(share["sd"], 4) if share else None,
                "kappa_young_mean": round(s["kappa_young"]["mean"], 2),
                "config": rec["config"],
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
    ap.add_argument("arm", nargs="+", choices=ARMS + ("all", "score", "frames"))
    ap.add_argument("--config", default="test", help="Sampling configuration (default: test).")
    ap.add_argument("--output-dir", required=True, help="Output root for this experiment; never a publication root.")
    args = ap.parse_args()

    from vocab_growth import environment as env
    from vocab_growth.analysis_frames import build_analysis_frame
    from vocab_growth.models.definitions import MODEL_REGISTRY

    output_root = os.path.abspath(args.output_dir)
    os.makedirs(output_root, exist_ok=True)
    env.set_output_root(output_root)

    base_definition = MODEL_REGISTRY[MODEL_KEY]
    vp = base_definition.subject_variance_partition
    assert vp is not None, "VG12 is expected to carry the variance partition."
    base_frame, _side = build_analysis_frame(MODEL_KEY, base_definition)
    frames = {"full": recode_dense(base_frame), "repeaters": repeaters_only(base_frame)}
    profiles = {k: replication_profile(v) for k, v in frames.items()}
    print(pd.DataFrame(profiles).T.to_string())
    registered_prior = (vp.share_alpha, vp.share_beta)
    print(f"registered share prior: Beta{registered_prior}")
    with open(os.path.join(output_root, "frames.json"), "w", encoding="utf-8") as fh:
        json.dump({"profiles": profiles, "registered_share_prior": registered_prior}, fh, indent=2)

    requested = list(args.arm)
    if "frames" in requested:
        return
    if "all" in requested:
        requested = list(ARMS)
    to_fit = [a for a in requested if a in ARMS]

    for arm in to_fit:
        if arm == "baseline":
            definition, frame, share_prior = base_definition, frames["full"], registered_prior
            calibration = None
        elif arm == "repeaters":
            definition, frame, share_prior = base_definition, frames["repeaters"], registered_prior
            calibration = None
        else:
            path = os.path.join(output_root, "repeaters.json")
            if not os.path.isfile(path):
                raise SystemExit(f"{arm} needs {path}; run the repeaters arm first.")
            with open(path, encoding="utf-8") as fh:
                rep = json.load(fh)["score"]["subject_variance_share"]
            scale = CONCENTRATION_SCALE[arm]
            definition, share_prior = calibrated_definition(
                base_definition, rep["mean"], rep["sd"], concentration_scale=scale
            )
            frame = frames["full"]
            calibration = {
                "from": "repeaters posterior on subject_variance_share",
                "posterior_mean": rep["mean"],
                "posterior_sd": rep["sd"],
                "concentration_scale": scale,
            }
        profile = replication_profile(frame)
        print(
            f"\n=== {arm}: {profile['rows']} rows, {profile['children']} children, "
            f"{100 * profile['repeater_share']:.1f}% repeaters, share prior Beta"
            f"({share_prior[0]:.3f}, {share_prior[1]:.3f}) ==="
        )
        trace, gate, trace_path, elapsed = fit_arm(definition, frame, args.config, output_root, arm)
        score = score_arm(trace, gate)
        record = {
            "arm": arm,
            "model": MODEL_KEY,
            "config": args.config,
            "frame": profile,
            "share_prior": [round(share_prior[0], 4), round(share_prior[1], 4)],
            "calibration": calibration,
            "score": score,
            "definition": dataclasses.asdict(definition),
            "trace": trace_path,
            "elapsed_seconds": elapsed,
        }
        with open(os.path.join(output_root, f"{arm}.json"), "w", encoding="utf-8") as fh:
            json.dump(record, fh, indent=2, default=str)
        share = score.get("subject_variance_share", {})
        print(
            f"{arm}: min BFMI {score['min_bfmi']:.3f}  corr(tau_subject, kappa_young) = "
            f"{score['ridge_tau_kappa_young']:+.3f}  corr(tau, energy) = {score['corr_tau_energy']:+.3f}  "
            f"divergences {score['divergences']}  max R-hat {score['max_rhat']}  "
            f"tau {score['tau_subject']['mean']:.4f} (sd {score['tau_subject']['sd']:.4f})  "
            + (f"share {share['mean']:.4f} (sd {share['sd']:.4f})  " if share else "")
            + f"({elapsed / 60:.1f} min)"
        )

    write_summary(output_root)


if __name__ == "__main__":
    freeze_support()
    main()
