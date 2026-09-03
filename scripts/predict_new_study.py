# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Score a model of record against a study it was never fitted to.

Usage:

    uv run python scripts/predict_new_study.py --frame <csv> --model vg20 \
        [--label us_03] [--draws 2000] [--out <dir>]

The published posterior of ``--model`` is used to predict every administration
in ``--frame``, which must **not** be a study the model was fitted on. Three
things are scored, in increasing order of what they test:

1. **Marginal prediction.** Each administration is predicted for an unseen child
   in an unseen study: the study offset is drawn from its fitted between-study
   scale and the child effects from theirs. Coverage of the 50% and 89%
   predictive intervals, and the probability integral transform, test the
   population curves and the two variance components together. Rows of one study
   share a study effect, so their errors are correlated and the effective number
   of independent tests is far below the row count -- which is what item 2 is for.

2. **Where the study sits.** The study's offset is estimated back out of the
   residuals and placed against the spread of the offsets the model fitted for
   the studies it did see. A new study landing inside that spread is the
   direct test of the reference-child estimand.

3. **Within-child prediction.** For children with two visits, the first visit is
   conditioned on and the second predicted. This is the only test here of the
   between-child scales and of the correlation between a child's comprehension
   standing and their conversion of comprehension into speech, which the fitted
   pool identifies largely through its own repeat visits. The study offset is
   absorbed into the child effect for this item, because both visits carry it and
   only their sum is identified from one visit.

Nothing is refitted. The frame is never written into ``data/``; registering it
there would change every Down syndrome prepared-frame hash and destroy the
out-of-sample property this script exists to exploit. See issue #288.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import UTC, datetime

import arviz as az
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import betabinom

from vocab_growth.environment import output_root, set_output_root
from vocab_growth.models.catalogue import CATALOGUE
from vocab_growth.models.definitions import MODEL_REGISTRY

EPSILON = 1e-6
AGE_BANDS = [0, 20, 24, 30, 36, 200]


def _model_dir(model_key: str, root: str) -> str:
    d = MODEL_REGISTRY[model_key]
    return os.path.join(root, "models", f"{d.model_id}-{d.config_name}")


def _interp_draws(x_plot: np.ndarray, values: np.ndarray, ages: np.ndarray) -> np.ndarray:
    """Interpolate ``values`` (draws x plot_id) onto ``ages``; returns draws x n."""
    out = np.empty((values.shape[0], ages.size), dtype=float)
    for i in range(values.shape[0]):
        out[i] = np.interp(ages, x_plot, values[i])
    return out


def _flat(posterior, name: str) -> np.ndarray:
    v = np.asarray(posterior[name])
    return v.reshape(-1, *v.shape[2:])


def _betabinom_draw(rng, n, p, kappa):
    p = np.clip(p, EPSILON, 1.0 - EPSILON)
    theta = rng.beta(p * kappa, (1.0 - p) * kappa)
    return rng.binomial(np.asarray(n, dtype=np.int64), theta)


def _pit(sample: np.ndarray, observed: float, rng) -> float:
    """Randomised PIT, which is uniform under a correct discrete predictive."""
    below = float(np.mean(sample < observed))
    equal = float(np.mean(sample == observed))
    return below + equal * float(rng.random())


def _coverage_table(df: pd.DataFrame, group: str) -> pd.DataFrame:
    rows = []
    for key, g in df.groupby(group, observed=True):
        rows.append(
            {
                group: str(key),
                "n": int(len(g)),
                "children": int(g["subject_id"].nunique()),
                "cover50": round(float(g["in50"].mean()), 3),
                "cover89": round(float(g["in89"].mean()), 3),
                "median_pit": round(float(g["pit"].median()), 3),
                "obs_median": round(float(g["observed"].median()), 1),
                "pred_median": round(float(g["pred_median"].median()), 1),
            }
        )
    return pd.DataFrame(rows)


def marginal_prediction(post, x_plot, frame, draws, rng, n_trials):
    """Item 1: predict every row for an unseen child in an unseen study."""
    ages = frame["age"].to_numpy(dtype=float)
    f_u = _interp_draws(x_plot, _flat(post, "f_u_plot")[draws], ages)
    h = _interp_draws(x_plot, _flat(post, "h_plot")[draws], ages)
    k_u = _interp_draws(x_plot, _flat(post, "kappa_u_plot")[draws], ages)
    k_s = _interp_draws(x_plot, _flat(post, "kappa_s_plot")[draws], ages)

    tau_u = _flat(post, "tau_u")[draws][:, None]
    tau_q = _flat(post, "tau_q")[draws][:, None]
    t_su = _flat(post, "tau_subj_u")[draws][:, None]
    t_sq = _flat(post, "tau_subj_q")[draws][:, None]
    rho = _flat(post, "rho_uq")[draws][:, None]

    n_draw = f_u.shape[0]
    codes, _ = pd.factorize(frame["subject_id"])
    n_child = int(codes.max()) + 1

    # One study effect per draw, shared by every row of the study.
    du_study = rng.standard_normal((n_draw, 1)) * tau_u
    dq_study = rng.standard_normal((n_draw, 1)) * tau_q

    # One correlated child-effect pair per child per draw.
    zu = rng.standard_normal((n_draw, n_child))
    zq = rng.standard_normal((n_draw, n_child))
    rho_c = np.clip(rho, -0.999, 0.999)
    du_child = (zu * t_su)[:, codes]
    dq_child = ((rho_c * zu + np.sqrt(1.0 - rho_c**2) * zq) * t_sq)[:, codes]

    p_u = expit(f_u + du_study + du_child)
    q = expit(h + dq_study + dq_child)

    y_u = _betabinom_draw(rng, n_trials, p_u, k_u)
    y_s_joint = _betabinom_draw(rng, y_u, q, k_s)
    # Spoken conditioned on the row's own observed comprehension isolates q.
    y_s_cond = _betabinom_draw(
        rng, np.broadcast_to(frame["understood"].to_numpy()[None, :], q.shape), q, k_s
    )

    out = []
    for name, sample, observed in (
        ("understood", y_u, frame["understood"].to_numpy()),
        ("spoken_joint", y_s_joint, frame["spoken"].to_numpy()),
        ("spoken_given_observed_understood", y_s_cond, frame["spoken"].to_numpy()),
    ):
        lo50, hi50 = np.percentile(sample, [25, 75], axis=0)
        lo89, hi89 = np.percentile(sample, [5.5, 94.5], axis=0)
        med = np.median(sample, axis=0)
        rec = pd.DataFrame(
            {
                "outcome": name,
                "subject_id": frame["subject_id"].to_numpy(),
                "timepoint": frame["timepoint"].to_numpy(),
                "age": frame["age"].to_numpy(),
                "observed": observed,
                "pred_median": med,
                "lo89": lo89,
                "hi89": hi89,
                "in50": (observed >= lo50) & (observed <= hi50),
                "in89": (observed >= lo89) & (observed <= hi89),
                "pit": [_pit(sample[:, i], observed[i], rng) for i in range(len(observed))],
                "pred_over_form": np.mean(sample > frame["survey_vocab_max"].to_numpy(), axis=0),
            }
        )
        out.append(rec)
    return pd.concat(out, ignore_index=True)


def study_offset(post, x_plot, frame, draws, n_trials, n_nodes=25):
    """Item 2: recover the study's own offset and place it against the fitted spread.

    A profile likelihood, not a moment estimate. The obvious estimator -- the mean
    of each row's empirical logit residual -- is unusable here: a study with a
    floor has rows at zero, ``logit`` is undefined there, and any continuity
    correction puts those rows at an extreme value that then dominates the mean.
    On ``us_03`` that estimator read -0.96 where 30 zero rows of 287 supplied more
    than half the displacement. Instead the offset is profiled over a grid, with
    the child effect integrated out by Gauss-Hermite quadrature, which handles a
    zero count as the ordinary Beta-Binomial event it is.
    """
    ages = frame["age"].to_numpy(dtype=float)
    f_u = _interp_draws(x_plot, _flat(post, "f_u_plot")[draws], ages).mean(axis=0)
    h = _interp_draws(x_plot, _flat(post, "h_plot")[draws], ages).mean(axis=0)
    k_u = _interp_draws(x_plot, _flat(post, "kappa_u_plot")[draws], ages).mean(axis=0)
    k_s = _interp_draws(x_plot, _flat(post, "kappa_s_plot")[draws], ages).mean(axis=0)
    t_su = float(_flat(post, "tau_subj_u")[draws].mean())
    t_sq = float(_flat(post, "tau_subj_q")[draws].mean())

    nodes, weights = np.polynomial.hermite_e.hermegauss(n_nodes)
    weights = weights / weights.sum()

    def profile(centre, kappa, counts, trials, tau):
        grid = np.arange(-2.5, 2.51, 0.02)
        ll = np.empty(grid.size)
        for j, delta in enumerate(grid):
            p = expit(centre[None, :] + delta + tau * nodes[:, None])
            p = np.clip(p, EPSILON, 1 - EPSILON)
            lp = betabinom.logpmf(
                counts[None, :], trials[None, :], p * kappa[None, :], (1 - p) * kappa[None, :]
            )
            ll[j] = np.sum(np.log(np.einsum("i,ij->j", weights, np.exp(lp)) + 1e-300))
        best = int(np.argmax(ll))
        # A 1.92 drop in log-likelihood is the usual 95% profile interval.
        keep = grid[ll >= ll[best] - 1.92]
        return float(grid[best]), float(keep.min()), float(keep.max())

    y_u = frame["understood"].to_numpy(dtype=np.int64)
    y_s = frame["spoken"].to_numpy(dtype=np.int64)
    trials_u = np.full(y_u.size, n_trials, dtype=np.int64)
    ok = y_u > 0

    fitted_u = _flat(post, "delta_u").mean(axis=0)
    fitted_q = _flat(post, "delta_q").mean(axis=0)

    rows = []
    for name, centre, kappa, counts, trials, tau, fitted in (
        ("understood", f_u, k_u, y_u, trials_u, t_su, fitted_u),
        ("production_ratio", h[ok], k_s[ok], y_s[ok], y_u[ok], t_sq, fitted_q),
    ):
        est, lo, hi = profile(centre, kappa, counts, trials, tau)
        rows.append(
            {
                "quantity": name,
                "n_rows": int(counts.size),
                "estimated_offset": round(est, 3),
                "ci95_lo": round(lo, 3),
                "ci95_hi": round(hi, 3),
                "fitted_studies_min": round(float(fitted.min()), 3),
                "fitted_studies_max": round(float(fitted.max()), 3),
                "fitted_studies_sd": round(float(fitted.std(ddof=1)), 3),
                "inside_fitted_range": bool(fitted.min() <= est <= fitted.max()),
                "n_fitted_studies_below": int((fitted < est).sum()),
            }
        )
    return pd.DataFrame(rows)


def within_child(post, x_plot, frame, draws, rng, n_trials, n_candidates=160):
    """Item 3: condition on a child's first visit and predict their second."""
    pairs = (
        frame[frame["timepoint"].isin(["t1", "t2"])]
        .sort_values(["subject_id", "age"])
        .groupby("subject_id")
        .filter(lambda g: len(g) == 2)
    )
    if pairs.empty:
        return pd.DataFrame(), pd.DataFrame()
    first = pairs.groupby("subject_id").first().reset_index()
    second = pairs.groupby("subject_id").last().reset_index()

    a1 = first["age"].to_numpy(dtype=float)
    a2 = second["age"].to_numpy(dtype=float)
    d = draws
    f1 = _interp_draws(x_plot, _flat(post, "f_u_plot")[d], a1)
    f2 = _interp_draws(x_plot, _flat(post, "f_u_plot")[d], a2)
    h1 = _interp_draws(x_plot, _flat(post, "h_plot")[d], a1)
    h2 = _interp_draws(x_plot, _flat(post, "h_plot")[d], a2)
    ku1 = _interp_draws(x_plot, _flat(post, "kappa_u_plot")[d], a1)
    ku2 = _interp_draws(x_plot, _flat(post, "kappa_u_plot")[d], a2)
    ks1 = _interp_draws(x_plot, _flat(post, "kappa_s_plot")[d], a1)
    ks2 = _interp_draws(x_plot, _flat(post, "kappa_s_plot")[d], a2)

    # The study offset is common to both visits and only its sum with the child
    # effect is identified from one visit, so it is absorbed here.
    su = np.sqrt(_flat(post, "tau_subj_u")[d] ** 2 + _flat(post, "tau_u")[d] ** 2)[:, None, None]
    sq = np.sqrt(_flat(post, "tau_subj_q")[d] ** 2 + _flat(post, "tau_q")[d] ** 2)[:, None, None]
    rho = np.clip(_flat(post, "rho_uq")[d], -0.999, 0.999)[:, None, None]

    nd, nc, nk = len(d), len(first), n_candidates
    zu = rng.standard_normal((nd, nc, nk))
    zq = rng.standard_normal((nd, nc, nk))
    cu = zu * su
    cq = (rho * zu + np.sqrt(1.0 - rho**2) * zq) * sq

    y_u1 = first["understood"].to_numpy()[None, :, None]
    y_s1 = first["spoken"].to_numpy()[None, :, None]
    p1 = np.clip(expit(f1[:, :, None] + cu), EPSILON, 1 - EPSILON)
    q1 = np.clip(expit(h1[:, :, None] + cq), EPSILON, 1 - EPSILON)
    k_u1 = ku1[:, :, None]
    k_s1 = ks1[:, :, None]

    logw = betabinom.logpmf(y_u1, n_trials, p1 * k_u1, (1 - p1) * k_u1)
    logw += betabinom.logpmf(y_s1, y_u1, q1 * k_s1, (1 - q1) * k_s1)
    logw -= logw.max(axis=2, keepdims=True)
    w = np.exp(logw)
    w /= w.sum(axis=2, keepdims=True)

    cum = np.cumsum(w, axis=2)
    pick = (cum < rng.random((nd, nc, 1))).sum(axis=2).clip(0, nk - 1)
    idx = np.take_along_axis(cu, pick[:, :, None], axis=2)[:, :, 0]
    idq = np.take_along_axis(cq, pick[:, :, None], axis=2)[:, :, 0]

    p2 = expit(f2 + idx)
    q2 = expit(h2 + idq)
    pu2 = _betabinom_draw(rng, n_trials, p2, ku2)
    ps2 = _betabinom_draw(rng, pu2, q2, ks2)
    ps2_cond = _betabinom_draw(
        rng, np.broadcast_to(second["understood"].to_numpy()[None, :], q2.shape), q2, ks2
    )

    out = []
    for name, t1_column, sample, observed in (
        ("understood", "understood", pu2, second["understood"].to_numpy()),
        ("spoken_joint", "spoken", ps2, second["spoken"].to_numpy()),
        ("spoken_given_observed_understood", "spoken", ps2_cond, second["spoken"].to_numpy()),
    ):
        lo50, hi50 = np.percentile(sample, [25, 75], axis=0)
        lo89, hi89 = np.percentile(sample, [5.5, 94.5], axis=0)
        out.append(
            pd.DataFrame(
                {
                    "outcome": name,
                    "subject_id": second["subject_id"].to_numpy(),
                    "age_t1": a1,
                    "age_t2": a2,
                    "observed_t1": first[t1_column].to_numpy(),
                    "observed": observed,
                    "pred_median": np.median(sample, axis=0),
                    "in50": (observed >= lo50) & (observed <= hi50),
                    "in89": (observed >= lo89) & (observed <= hi89),
                    "pit": [_pit(sample[:, i], observed[i], rng) for i in range(len(observed))],
                }
            )
        )
    return pd.concat(out, ignore_index=True), first


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frame", required=True, help="CSV of the unseen study")
    ap.add_argument("--model", default="vg20")
    ap.add_argument("--label", default=None, help="name for the output directory")
    ap.add_argument("--draws", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260903)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.model not in CATALOGUE:
        raise SystemExit(f"unknown model {args.model!r}")
    set_output_root(args.out)
    root = output_root()
    mdir = _model_dir(args.model, root)
    definition = MODEL_REGISTRY[args.model]
    n_trials = definition.n_trials

    frame = pd.read_csv(args.frame)
    kept = frame[frame["understood"] <= frame["survey_vocab_max"]].copy()
    over_form = len(frame) - len(kept)
    # Both outcomes must be present. The fitted Down syndrome pool carries 448
    # spoken-only rows on a fallback likelihood; a frame holding any would
    # otherwise be compared against NaN, which silently reads as a miss and
    # deflates every coverage figure. Scoring the fallback branch is separate work.
    complete = kept["understood"].notna() & kept["spoken"].notna()
    incomplete = int((~complete).sum())
    kept = kept[complete].copy()
    kept["understood"] = kept["understood"].astype(np.int64)
    kept["spoken"] = kept["spoken"].astype(np.int64)
    if "timepoint" not in kept:
        kept["timepoint"] = "t1"
    print(
        f"[frame] {args.frame}: {len(frame)} rows; dropped {over_form} exceeding the form "
        f"and {incomplete} missing an outcome"
    )
    print(f"[frame] scoring {len(kept)} rows, {kept['subject_id'].nunique()} children")

    tree = az.from_netcdf(os.path.join(mdir, "trace.nc"))
    post = tree["posterior"]
    x_plot = np.asarray(tree["constant_data"]["X_plot"]).ravel()
    total = int(np.asarray(post["tau_u"]).size)
    rng = np.random.default_rng(args.seed)
    draws = rng.choice(total, size=min(args.draws, total), replace=False)
    print(f"[trace] {mdir}: {total} draws, using {draws.size}")

    label = args.label or os.path.splitext(os.path.basename(args.frame))[0]
    odir = os.path.join(root, "comparisons", "oos", label)
    os.makedirs(odir, exist_ok=True)

    marg = marginal_prediction(post, x_plot, kept, draws, rng, n_trials)
    marg.to_csv(os.path.join(odir, f"oos_marginal_{args.model}.csv"), index=False)
    marg["age_band"] = pd.cut(marg["age"], AGE_BANDS, right=False)
    cov = pd.concat(
        [
            _coverage_table(g.assign(outcome=k), "outcome").assign(scope="all")
            for k, g in marg.groupby("outcome", observed=True)
        ]
        + [
            _coverage_table(g, "age_band").assign(scope=k)
            for k, g in marg.groupby("outcome", observed=True)
        ],
        ignore_index=True,
    )
    cov.to_csv(os.path.join(odir, f"oos_coverage_{args.model}.csv"), index=False)

    off = study_offset(post, x_plot, kept, draws, n_trials)
    off.to_csv(os.path.join(odir, f"oos_study_offset_{args.model}.csv"), index=False)

    cond, _ = within_child(post, x_plot, kept, draws, rng, n_trials)
    if not cond.empty:
        cond.to_csv(os.path.join(odir, f"oos_within_child_{args.model}.csv"), index=False)
        cond_cov = _coverage_table(cond.assign(subject_id=cond["subject_id"]), "outcome")
        cond_cov.to_csv(os.path.join(odir, f"oos_within_child_coverage_{args.model}.csv"), index=False)

    manifest = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "model": args.model,
        "model_dir": mdir,
        "frame": os.path.abspath(args.frame),
        "frame_sha256": hashlib.sha256(open(args.frame, "rb").read()).hexdigest(),
        "rows_in_file": int(len(frame)),
        "rows_dropped_exceeding_form": int(over_form),
        "rows_dropped_missing_outcome": int(incomplete),
        "rows_scored": int(len(kept)),
        "children": int(kept["subject_id"].nunique()),
        "posterior_draws_used": int(draws.size),
        "seed": args.seed,
        "n_trials": int(n_trials),
    }
    fit_manifest = os.path.join(mdir, "fit_manifest.json")
    if os.path.exists(fit_manifest):
        with open(fit_manifest, encoding="utf-8") as fh:
            fm = json.load(fh)
        manifest["contributing_fit"] = {
            "analysis_frame_hash": fm.get("data", {}).get("analysis_frame_hash"),
            "sampling_config": fm.get("sampling", {}).get("config_name"),
        }
    with open(os.path.join(odir, f"oos_manifest_{args.model}.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    print(f"\n[written] {odir}")
    print("\n=== marginal coverage (nominal 0.50 / 0.89)")
    print(cov[cov["scope"] == "all"].to_string(index=False))
    print("\n=== study offset against the fitted spread")
    print(off.to_string(index=False))
    if not cond.empty:
        print("\n=== second visit predicted from the first")
        print(cond_cov.to_string(index=False))


if __name__ == "__main__":
    main()
