# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Controls, age profiles and null distributions for ``predict_new_study.py``.

The scores that script writes only mean something against a reference: what the
machinery returns on data the model *did* see, how far its offset estimator can
be trusted, and what a study that the model itself generated would look like
under the same scoring. This script produces those references and writes them
next to the scores, so the note that reads them is regenerable end to end.

    uv run python scripts/predict_new_study_checks.py controls --model vg20
    uv run python scripts/predict_new_study_checks.py profile --frame <csv> \
        --models vg02,vg09,vg10,vg19,vg20,vg22 --bands 17-19,20-25,29-32,33-35
    uv run python scripts/predict_new_study_checks.py null --frame <csv> \
        --model vg20 --reps 25 --part 1 --bands 17-19,20-25,29-32,33-35
    uv run python scripts/predict_new_study_checks.py null-summary --model vg20
    uv run python scripts/predict_new_study_checks.py collate --reference vg20

``controls`` predicts the model's own fitted rows as if they were an unseen
study and unseen children (coverage must then sit at or above nominal), and
recovers the fitted per-study offsets with the profile estimator.

``profile`` estimates the study offset on comprehension within age bands, within
visits, and separately for children with one and two visits. Several bandings
may be given; the first is the primary one and the rest are there to show how
much the band-to-band spread depends on where the cuts fall. For each banding
it reports the spread (largest minus smallest band offset) and a likelihood
ratio statistic, twice the gain in profile log-likelihood from letting each
band have its own offset, which does not depend on which single band is most
extreme. The ratio is only reported for models whose child effect is constant
in age; for a slope or factor structure the per-band profiles are not nested
in the whole-frame one. For children with two visits it reports the offset at each visit and
the change between them, which is the within-child test of whether the
departure grows as a child ages.

``null`` simulates studies from the model with the frame's own ages, children
and visit structure -- one posterior draw as the truth, a study offset drawn at
the fitted between-study scale, child effects shared across a child's visits,
counts censored at the form length -- and scores each with the same code. A
simulated study has a constant offset by construction, so the spread and
likelihood-ratio statistics it returns measure estimation noise alone; that is
the reference distribution for the corresponding statistics on the real study.

``null`` is single-threaded; run it as several ``--part``s with different
``--seed``s in parallel and pool them with ``null-summary``, which also prints
where the real study's statistics fall in the pooled distribution.

``collate`` gathers the per-model scores into one table and computes paired
differences in second-visit log predictive density against a reference model.
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import os
from datetime import UTC, datetime

import numpy as np
import pandas as pd
from scipy.special import expit

from vocab_growth.analysis_frames import build_analysis_frame
from vocab_growth.environment import output_root, set_output_root
from vocab_growth.models.definitions import MODEL_REGISTRY

_SPEC = importlib.util.spec_from_file_location(
    "predict_new_study", os.path.join(os.path.dirname(__file__), "predict_new_study.py")
)
pns = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(pns)


def _parse_bands(spec: str) -> list[tuple[int, int]]:
    """``"17-19,20-25"`` -> ``[(17, 20), (20, 26)]``: inclusive months to half-open."""
    bands = []
    for part in spec.split(","):
        lo, hi = part.split("-")
        bands.append((int(lo), int(hi) + 1))
    return bands


def _band_label(lo: int, hi: int) -> str:
    return f"{lo}-{hi - 1}"


def _offset(post, x_plot, frame, draws, n_trials, definition):
    row = pns.study_offset(
        post, x_plot, frame.reset_index(drop=True), draws, n_trials, definition
    ).iloc[0]
    return (
        float(row["estimated_offset"]),
        float(row["ci95_lo"]),
        float(row["ci95_hi"]),
        float(row["max_loglik"]),
    )


def offset_profile(post, x_plot, frame, draws, n_trials, definition, schemes, max_age):
    """Comprehension offset by band, by visit and by pairing, as one flat record."""
    out: dict = {}
    # The profile integrates the child effect at a scale taken at each
    # sub-frame's median age. For a constant child structure that scale is the
    # same in every band, so the band likelihoods nest in the whole-frame one
    # and the ratio is a proper statistic; for a slope or factor structure it
    # is not, and the ratio is withheld rather than reported as if it were.
    nested = pns.child_structure(post, definition) in ("constant", "none")
    est, lo, hi, ll_whole = _offset(post, x_plot, frame, draws, n_trials, definition)
    out.update({"whole": est, "whole_lo": lo, "whole_hi": hi})
    for scheme in schemes:
        bands = _parse_bands(scheme)
        ests, ll_sum = [], 0.0
        for b_lo, b_hi in bands:
            g = frame[(frame["age"] >= b_lo) & (frame["age"] < b_hi)]
            e, _, _, ll = _offset(post, x_plot, g, draws, n_trials, definition)
            out[f"band[{_band_label(b_lo, b_hi)}]"] = e
            out[f"n[{_band_label(b_lo, b_hi)}]"] = int(len(g))
            ests.append(e)
            ll_sum += ll
        inside = frame["age"].between(bands[0][0], bands[-1][1] - 1)
        if not inside.all():
            # The whole-frame likelihood is not comparable when rows fall outside
            # every band; profile the banded rows jointly instead.
            _, _, _, ll_whole_b = _offset(
                post, x_plot, frame[inside], draws, n_trials, definition
            )
        else:
            ll_whole_b = ll_whole
        out[f"spread[{scheme}]"] = round(max(ests) - min(ests), 3)
        out[f"lr[{scheme}]"] = (
            round(2.0 * (ll_sum - ll_whole_b), 2) if nested else float("nan")
        )

    visits = frame.groupby("subject_id").size()
    paired = frame[frame["subject_id"].isin(visits[visits == 2].index)]
    single = frame[frame["subject_id"].isin(visits[visits == 1].index)]
    if max_age is not None:
        single = single[single["age"] < max_age]
    if len(paired):
        t1 = paired[paired["timepoint"] == "t1"]
        t2 = paired[paired["timepoint"] == "t2"]
        e1, lo1, hi1, _ = _offset(post, x_plot, t1, draws, n_trials, definition)
        e2, lo2, hi2, _ = _offset(post, x_plot, t2, draws, n_trials, definition)
        out.update(
            {
                "n_paired": int(len(t1)),
                "t1_paired": e1,
                "t1_paired_lo": lo1,
                "t1_paired_hi": hi1,
                "t1_paired_age_median": float(t1["age"].median()),
                "t2_paired": e2,
                "t2_paired_lo": lo2,
                "t2_paired_hi": hi2,
                "t2_paired_age_median": float(t2["age"].median()),
                "paired_change": round(e2 - e1, 3),
            }
        )
    if len(single):
        e, lo, hi, _ = _offset(post, x_plot, single, draws, n_trials, definition)
        out.update(
            {
                "n_single": int(len(single)),
                "t1_single": e,
                "t1_single_lo": lo,
                "t1_single_hi": hi,
                "t1_single_age_median": float(single["age"].median()),
            }
        )
    return out


def _reference_curve(post, x_plot, prof, ages):
    f = pns._flat(post, prof["f"]).mean(axis=0)
    return expit(np.interp(ages, x_plot, f))


def fine_profile(post, x_plot, frame, draws, n_trials, definition, min_rows=5):
    """The offset at every single month of age that carries ``min_rows`` rows."""
    prof = pns.engine_profile(post)
    rows = []
    for age, g in frame.groupby("age"):
        if len(g) < min_rows:
            continue
        e, lo, hi, _ = _offset(post, x_plot, g, draws, n_trials, definition)
        rows.append(
            {
                "age": int(age),
                "n": int(len(g)),
                "timepoints": ",".join(sorted(g["timepoint"].unique())),
                "offset": e,
                "ci95_lo": lo,
                "ci95_hi": hi,
                "observed_median_understood": float(g["understood"].median()),
                "reference_child_understood": round(
                    float(
                        n_trials
                        * _reference_curve(post, x_plot, prof, np.array([age]))[0]
                    ),
                    1,
                ),
            }
        )
    return pd.DataFrame(rows)


def _out_dir(root: str, label: str) -> str:
    odir = os.path.join(root, "comparisons", "oos", label)
    os.makedirs(odir, exist_ok=True)
    return odir


def _draws(rng, total, n):
    return rng.choice(total, size=min(n, total), replace=False)


# --------------------------------------------------------------------------- controls


def run_controls(args, root):
    definition = MODEL_REGISTRY[args.model]
    n_trials = definition.n_trials
    mdir, post, x_plot, total = pns.load_posterior(args.model, root)
    rng = np.random.default_rng(args.seed)
    draws = _draws(rng, total, args.draws)
    odir = _out_dir(root, args.label)

    frame, _ = build_analysis_frame(args.model, definition)
    frame = frame.copy()
    n_all = len(frame)
    complete = frame["understood"].notna() & frame["spoken"].notna()
    frame = frame[complete].copy()
    n_complete = len(frame)
    n_inverted = int((frame["spoken"] > frame["understood"]).sum())
    frame = frame[frame["spoken"] <= frame["understood"]].copy()
    frame["understood"] = frame["understood"].astype(np.int64)
    frame["spoken"] = frame["spoken"].astype(np.int64)
    frame["subject_id"] = frame["subject_key"]
    frame["timepoint"] = "t1"
    frame = frame.reset_index(drop=True)
    print(
        f"[frame] {args.model}: {n_all} fitted rows, {n_complete} with both outcomes, "
        f"{n_inverted} with spoken above understood dropped, {len(frame)} scored"
    )

    marg = pns.marginal_prediction(
        post, x_plot, frame, draws, rng, n_trials, definition
    )
    cal = pns._coverage_table(marg, "outcome")
    cal.insert(0, "model", args.model)
    cal["rows_scored"] = len(frame)
    cal.to_csv(
        os.path.join(odir, f"oos_control_calibration_{args.model}.csv"), index=False
    )
    print("\n=== fitted rows predicted as an unseen study and unseen children")
    print(cal.to_string(index=False))

    rows = []
    fitted = {
        name: pns._flat(post, name).mean(axis=0)
        for name in ("delta_u", "delta_q")
        if name in post.data_vars
    }
    for study, g in frame.groupby("study"):
        if len(g) < args.min_rows:
            continue
        code = int(g["study_code"].iloc[0])
        est = pns.study_offset(
            post, x_plot, g.reset_index(drop=True), draws, n_trials, definition
        )
        row = {"study": study, "n_rows": int(len(g))}
        for quantity, name in (
            ("understood", "delta_u"),
            ("production_ratio", "delta_q"),
        ):
            sel = est[est["quantity"] == quantity]
            if sel.empty or name not in fitted:
                continue
            row[f"fitted_{quantity}"] = round(float(fitted[name][code]), 3)
            row[f"estimated_{quantity}"] = float(sel["estimated_offset"].iloc[0])
        rows.append(row)
    rec = pd.DataFrame(rows)
    rec.to_csv(
        os.path.join(odir, f"oos_control_fitted_offsets_{args.model}.csv"), index=False
    )
    print(
        f"\n=== profile estimator against the fitted offsets (studies with >= {args.min_rows} rows)"
    )
    print(rec.to_string(index=False))
    for quantity in ("understood", "production_ratio"):
        a, b = f"fitted_{quantity}", f"estimated_{quantity}"
        if a in rec and b in rec:
            err = rec[b] - rec[a]
            big = rec[rec["n_rows"] >= 100]
            big_err = (big[b] - big[a]).abs().max() if len(big) else float("nan")
            print(
                f"  {quantity}: corr {np.corrcoef(rec[a], rec[b])[0, 1]:.3f}  "
                f"mean error {err.mean():+.3f}  max |error| {err.abs().max():.3f}  "
                f"max |error| at n>=100 {big_err:.3f}"
            )
    _write_manifest(odir, "controls", args, {args.model: mdir}, draws.size)


# ---------------------------------------------------------------------------- profile


def run_profile(args, root):
    frame, counts = pns.load_frame(args.frame)
    odir = _out_dir(root, args.label)
    rng = np.random.default_rng(args.seed)
    records, fine, contributing = [], [], {}
    for key in args.models.split(","):
        definition = MODEL_REGISTRY[key]
        mdir, post, x_plot, total = pns.load_posterior(key, root)
        contributing[key] = mdir
        draws = _draws(rng, total, args.draws)
        rec = offset_profile(
            post,
            x_plot,
            frame,
            draws,
            definition.n_trials,
            definition,
            args.bands,
            args.max_age,
        )
        rec = {
            "model": key.upper(),
            "structure": pns.child_structure(post, definition),
            **rec,
        }
        records.append(rec)
        f = fine_profile(post, x_plot, frame, draws, definition.n_trials, definition)
        f.insert(0, "model", key.upper())
        fine.append(f)
        print(
            f"[profile] {key}: whole {rec['whole']:+.2f}; "
            + "; ".join(
                f"{k} {v:+.2f}" for k, v in rec.items() if k.startswith("band[")
            )
        )
    table = pd.DataFrame(records)
    table.to_csv(os.path.join(odir, "oos_offset_profile.csv"), index=False)
    pd.concat(fine, ignore_index=True).to_csv(
        os.path.join(odir, "oos_offset_by_month.csv"), index=False
    )
    print("\n=== comprehension offset by age band, visit and pairing")
    print(table.T.to_string())
    _write_manifest(odir, "profile", args, contributing, args.draws, counts)


# ------------------------------------------------------------------------------- null


def simulate_study(post, x_plot, frame, definition, truth_draw, rng, n_trials):
    """One study drawn from the model with ``frame``'s ages, children and visits."""
    prof = pns.engine_profile(post)
    structure = pns.child_structure(post, definition)
    ref = pns.child_ref_age(structure, definition)
    ages = frame["age"].to_numpy(dtype=float)
    codes, _ = pd.factorize(frame["subject_id"])
    n_child = int(codes.max()) + 1
    d = np.array([truth_draw])
    curve = {
        k: np.interp(ages, x_plot, pns._flat(post, prof[k])[truth_draw])
        for k in ("f", "h", "ku", "ks")
    }
    tau_u = float(pns._flat(post, prof["tau_u"])[truth_draw])
    tau_q = float(pns._flat(post, prof["tau_q"])[truth_draw])
    offset_u, offset_q = rng.normal(0.0, tau_u), rng.normal(0.0, tau_q)
    params = pns.draw_child_params(post, definition, structure, d, rng, (1, n_child))[0]
    du, dq = pns.child_deltas(params[codes], ages, ref)
    p = expit(curve["f"] + offset_u + du)
    q = expit(curve["h"] + offset_q + dq)
    y_u = pns._betabinom_draw(rng, n_trials, p, curve["ku"])
    form = frame["survey_vocab_max"].to_numpy()
    censored = int((y_u > form).sum())
    y_u = np.minimum(y_u, form)
    y_s = pns._betabinom_draw(rng, y_u, q, curve["ks"])
    sim = frame.copy()
    sim["understood"] = y_u.astype(np.int64)
    sim["spoken"] = y_s.astype(np.int64)
    return sim, offset_u, offset_q, censored


def run_null(args, root):
    definition = MODEL_REGISTRY[args.model]
    n_trials = definition.n_trials
    frame, counts = pns.load_frame(args.frame)
    mdir, post, x_plot, total = pns.load_posterior(args.model, root)
    if not pns.engine_profile(post)["bivariate"]:
        raise SystemExit("the null simulation needs a bivariate model")
    odir = _out_dir(root, args.label)
    rng = np.random.default_rng(args.seed)
    rows = []
    for rep in range(args.reps):
        truth = int(rng.integers(0, total))
        sim, off_u, off_q, censored = simulate_study(
            post, x_plot, frame, definition, truth, rng, n_trials
        )
        draws = _draws(rng, total, args.draws)
        marg = pns.marginal_prediction(
            post, x_plot, sim, draws, rng, n_trials, definition
        )
        u = marg[marg["outcome"] == "understood"]
        s = marg[marg["outcome"] == "spoken_given_observed_understood"]
        off = pns.study_offset(post, x_plot, sim, draws, n_trials, definition)
        rec = {
            "replicate": rep,
            "truth_draw": truth,
            "true_offset_u": round(off_u, 3),
            "true_offset_q": round(off_q, 3),
            "rows_censored_at_form": censored,
            "cover50_u": round(float(u["in50"].mean()), 3),
            "cover89_u": round(float(u["in89"].mean()), 3),
            "median_pit_u": round(float(u["pit"].median()), 3),
            "cover50_s": round(float(s["in50"].mean()), 3),
            "cover89_s": round(float(s["in89"].mean()), 3),
            "median_pit_s": round(float(s["pit"].median()), 3),
            "estimated_offset_u": float(off.loc[0, "estimated_offset"]),
            "estimated_offset_q": float(off.loc[1, "estimated_offset"]),
        }
        rec.update(
            offset_profile(
                post,
                x_plot,
                sim,
                _draws(rng, total, args.profile_draws),
                n_trials,
                definition,
                args.bands,
                args.max_age,
            )
        )
        rows.append(rec)
        print(
            f"[null] {rep + 1}/{args.reps}: offset {off_u:+.2f} -> est {rec['estimated_offset_u']:+.2f}, "
            f"cover89_u {rec['cover89_u']:.3f}, spread {rec[f'spread[{args.bands[0]}]']:.2f}, "
            f"paired change {rec.get('paired_change', float('nan')):+.2f}"
        )
    table = pd.DataFrame(rows)
    suffix = f"_part{args.part}" if args.part is not None else ""
    table.to_csv(os.path.join(odir, f"oos_null_{args.model}{suffix}.csv"), index=False)
    _summarise_null(table, odir, args)
    _write_manifest(odir, f"null{suffix}", args, {args.model: mdir}, args.draws, counts)


def run_null_summary(args, root):
    """Pool the parts of a null run (or read a single run) and summarise it."""
    odir = _out_dir(root, args.label)
    parts = sorted(glob.glob(os.path.join(odir, f"oos_null_{args.model}_part*.csv")))
    if parts:
        table = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
        table["replicate"] = np.arange(len(table))
        table.to_csv(os.path.join(odir, f"oos_null_{args.model}.csv"), index=False)
        print(f"[null] pooled {len(parts)} parts into {len(table)} replicates")
    else:
        table = pd.read_csv(os.path.join(odir, f"oos_null_{args.model}.csv"))
    _summarise_null(table, odir, args)


def _summarise_null(table, odir, args):
    err_u = table["estimated_offset_u"] - table["true_offset_u"]
    print(
        f"\n=== {len(table)} studies simulated from {args.model} with the frame's structure"
    )
    print(
        f"offset recovery, understood: mean error {err_u.mean():+.3f}, sd {err_u.std(ddof=1):.3f}, "
        f"max |error| {err_u.abs().max():.3f}"
    )
    print(
        f"rows censored at the form, mean {table['rows_censored_at_form'].mean():.1f}"
    )
    stats = (
        ["cover89_u", "median_pit_u", "cover89_s"]
        + [c for c in table.columns if c.startswith(("spread[", "lr["))]
        + ["paired_change"]
    )
    stats = [c for c in stats if c in table]
    q = table[stats].quantile([0.0, 0.05, 0.5, 0.95, 1.0]).T
    q.columns = ["min", "q05", "median", "q95", "max"]
    print(q.round(3).to_string())
    real_path = os.path.join(odir, "oos_offset_profile.csv")
    cov_path = os.path.join(odir, f"oos_coverage_{args.model}.csv")
    if os.path.exists(real_path):
        real = pd.read_csv(real_path)
        real = real[real["model"] == args.model.upper()]
        if len(real):
            print("\n=== the real study against the null")
            for c in stats:
                if c in real and c not in ("cover89_u", "median_pit_u", "cover89_s"):
                    v = float(real[c].iloc[0])
                    tail = (
                        (table[c] >= v).sum()
                        if not c.startswith("paired")
                        else (table[c] <= v).sum()
                    )
                    print(
                        f"  {c}: real {v:+.3f}; simulated at or beyond it {tail} of {len(table)}"
                    )
    if os.path.exists(cov_path):
        cov = pd.read_csv(cov_path)
        cov = cov[cov["scope"] == "all"]
        v = float(cov[cov["outcome"] == "understood"]["cover89"].iloc[0])
        print(
            f"  cover89_u: real {v:.3f}; simulated at or below it {(table['cover89_u'] <= v).sum()} of {len(table)}"
        )
        near = (
            table[
                (table["true_offset_u"] - float(real["whole"].iloc[0])).abs()
                <= args.offset_window
            ]
            if os.path.exists(real_path) and len(real)
            else table.iloc[0:0]
        )
        if len(near):
            print(
                f"  simulated studies within {args.offset_window} of the real offset (n={len(near)}): "
                f"cover89_u {sorted(near['cover89_u'].round(3).tolist())}"
            )


# ---------------------------------------------------------------------------- collate


def run_collate(args, root):
    odir = _out_dir(root, args.label)
    rows = []
    for path in sorted(glob.glob(os.path.join(odir, "oos_coverage_vg*.csv"))):
        key = os.path.basename(path)[len("oos_coverage_") : -4]
        cov = pd.read_csv(path)
        cov = cov[cov["scope"] == "all"]
        off = pd.read_csv(path.replace("oos_coverage_", "oos_study_offset_"))
        u = cov[cov["outcome"] == "understood"].iloc[0]
        row = {
            "model": key.upper(),
            "marginal_cover89_understood": u["cover89"],
            "marginal_median_pit_understood": u["median_pit"],
            "offset_understood": off.loc[0, "estimated_offset"],
        }
        s = cov[cov["outcome"] == "spoken_given_observed_understood"]
        if len(s):
            row["marginal_cover89_spoken_given_understood"] = s.iloc[0]["cover89"]
            row["offset_production_ratio"] = off.loc[1, "estimated_offset"]
        wpath = path.replace("oos_coverage_", "oos_within_child_coverage_")
        if os.path.exists(wpath):
            w = pd.read_csv(wpath)
            ws = w[w["outcome"] == "spoken_given_observed_understood"].iloc[0]
            wu = w[w["outcome"] == "understood"].iloc[0]
            row.update(
                {
                    "second_visit_cover89_spoken_given_understood": ws["cover89"],
                    "second_visit_median_pit_spoken_given_understood": ws["median_pit"],
                    "second_visit_cover89_understood": wu["cover89"],
                    "second_visit_median_pit_understood": wu["median_pit"],
                }
            )
        lpath = path.replace("oos_coverage_", "oos_within_child_lpd_")
        if os.path.exists(lpath):
            lpd = pd.read_csv(lpath)
            for c in pns.LPD_COLUMNS:
                row[f"total_{c}"] = round(float(lpd[c].sum()), 2)
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(odir, "oos_model_comparison.csv"), index=False)
    print("=== every scored model")
    print(table.T.to_string())

    ref_path = os.path.join(odir, f"oos_within_child_lpd_{args.reference}.csv")
    if not os.path.exists(ref_path):
        return
    ref = pd.read_csv(ref_path)
    diffs = []
    for path in sorted(glob.glob(os.path.join(odir, "oos_within_child_lpd_vg*.csv"))):
        key = os.path.basename(path)[len("oos_within_child_lpd_") : -4]
        if key == args.reference:
            continue
        other = pd.read_csv(path)
        merged = ref.merge(other, on="subject_id", suffixes=("_ref", "_other"))
        for c in pns.LPD_COLUMNS:
            d = merged[f"{c}_ref"] - merged[f"{c}_other"]
            se = d.std(ddof=1) / np.sqrt(len(d))
            diffs.append(
                {
                    "comparison": f"{args.reference.upper()} minus {key.upper()}",
                    "quantity": c[len("lpd_") :],
                    "children": int(len(d)),
                    "total": round(float(d.sum()), 2),
                    "per_child": round(float(d.mean()), 4),
                    "se": round(float(se), 4),
                    "z": round(float(d.mean() / se), 2),
                }
            )
    # Within each model, what adding the visit-1 production count to the
    # conditioning does to the visit-2 comprehension prediction. Only a model
    # that couples the two child effects can move here; the split by the
    # child's visit-1 production ratio says where any movement comes from.
    for path in sorted(glob.glob(os.path.join(odir, "oos_within_child_lpd_vg*.csv"))):
        key = os.path.basename(path)[len("oos_within_child_lpd_") : -4]
        lpd = pd.read_csv(path)
        wc = pd.read_csv(path.replace("oos_within_child_lpd_", "oos_within_child_"))
        t1 = wc[wc["outcome"] == "understood"][["subject_id", "observed_t1"]].merge(
            wc[wc["outcome"] == "spoken_joint"][["subject_id", "observed_t1"]],
            on="subject_id",
            suffixes=("_u", "_s"),
        )
        t1["ratio_t1"] = t1["observed_t1_s"] / t1["observed_t1_u"].clip(lower=1)
        lpd = lpd.merge(t1[["subject_id", "ratio_t1"]], on="subject_id")
        gain = lpd["lpd_understood_given_both"] - lpd["lpd_understood_given_understood"]
        for label, mask in (
            ("all children", np.ones(len(lpd), dtype=bool)),
            ("visit-1 production ratio >= 0.2", lpd["ratio_t1"] >= 0.2),
            ("visit-1 production ratio < 0.2", lpd["ratio_t1"] < 0.2),
        ):
            g = gain[mask]
            se = g.std(ddof=1) / np.sqrt(len(g)) if len(g) > 1 else float("nan")
            diffs.append(
                {
                    "comparison": f"{key.upper()}: given both minus given comprehension only",
                    "quantity": f"understood, {label}",
                    "children": int(len(g)),
                    "total": round(float(g.sum()), 2),
                    "per_child": round(float(g.mean()), 4),
                    "se": round(float(se), 4),
                    "z": round(float(g.mean() / se), 2) if se == se else float("nan"),
                }
            )
    diff = pd.DataFrame(diffs)
    diff.to_csv(os.path.join(odir, "oos_lpd_comparison.csv"), index=False)
    print(
        f"\n=== second-visit log predictive density, paired against {args.reference.upper()}"
    )
    print(diff.to_string(index=False))


# --------------------------------------------------------------------------- manifest


def _write_manifest(odir, command, args, contributing, draws, counts=None):
    from vocab_growth.comparisons_provenance import fit_manifest_fingerprint

    payload = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "command": command,
        "arguments": {k: v for k, v in vars(args).items() if k != "func"},
        "posterior_draws_used": int(draws),
        "contributing_fits": {
            k: fit_manifest_fingerprint(v) for k, v in contributing.items()
        },
    }
    if counts:
        payload["frame"] = counts
    with open(
        os.path.join(odir, f"oos_checks_manifest_{command}.json"), "w", encoding="utf-8"
    ) as fh:
        json.dump(payload, fh, indent=2)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", default=None, help="output root (see fit_model.py)")
    ap.add_argument(
        "--label", default="us_03", help="output directory under comparisons/oos/"
    )
    ap.add_argument("--seed", type=int, default=20260903)
    sub = ap.add_subparsers(dest="command", required=True)

    c = sub.add_parser("controls")
    c.add_argument("--model", default="vg20")
    c.add_argument("--draws", type=int, default=900)
    c.add_argument("--min-rows", type=int, default=25)
    c.set_defaults(func=run_controls)

    p = sub.add_parser("profile")
    p.add_argument("--frame", required=True)
    p.add_argument("--models", default="vg02,vg09,vg10,vg19,vg20,vg22")
    p.add_argument(
        "--bands",
        action="append",
        required=True,
        help="e.g. 17-19,20-25,29-32,33-35; repeatable",
    )
    p.add_argument(
        "--max-age", type=float, default=None, help="cap for the single-visit group"
    )
    p.add_argument("--draws", type=int, default=900)
    p.set_defaults(func=run_profile)

    n = sub.add_parser("null")
    n.add_argument("--frame", required=True)
    n.add_argument("--model", default="vg20")
    n.add_argument("--reps", type=int, default=30)
    n.add_argument("--bands", action="append", required=True)
    n.add_argument("--max-age", type=float, default=None)
    n.add_argument(
        "--draws",
        type=int,
        default=1500,
        help="posterior draws for the marginal predictive",
    )
    n.add_argument("--profile-draws", type=int, default=900)
    n.add_argument("--offset-window", type=float, default=0.15)
    n.add_argument(
        "--part",
        type=int,
        default=None,
        help="write oos_null_<model>_part<N>.csv, for parallel runs with different seeds",
    )
    n.set_defaults(func=run_null)

    m = sub.add_parser("null-summary")
    m.add_argument("--model", default="vg20")
    m.add_argument("--offset-window", type=float, default=0.15)
    m.set_defaults(func=run_null_summary)

    k = sub.add_parser("collate")
    k.add_argument("--reference", default="vg20")
    k.set_defaults(func=run_collate)

    args = ap.parse_args()
    set_output_root(args.out)
    args.func(args, output_root())


if __name__ == "__main__":
    main()
