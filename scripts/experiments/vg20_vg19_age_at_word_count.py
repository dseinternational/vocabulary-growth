# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Age by which a share of DS children reaches a word count -- VG20 vs VG19.

Same estimand as ``age_at_word_count.py``, computed under both child-effect
structures so the tails can be compared:

* **VG20** (model of record) gives each child one constant offset per outcome,
  ``z_u`` on the logit of understood and ``z_q`` on the logit of the production
  ratio, correlated across outcomes by ``rho_uq``. The between-child band is the
  same width at every age.
* **VG19** gives each child an intercept *and* a rate per outcome, drawn from a
  per-outcome 2x2 covariance: the effect at age ``a`` is ``b0 + b1 * D(a)`` with
  ``D(a) = (a - 36) / 12``. Within an outcome the two are correlated; across
  outcomes VG19 forces independence, which is exactly the axis VG20 models.

Neither model is nested in the other -- VG20 correlates the outcomes but freezes
the spread in age, VG19 grows the spread but decouples the outcomes -- so this is
a sensitivity comparison, not a refinement. VG19 is not the model of record.

Both are run through the *same* simulator, which doubles as a check: VG20's
understood column has an exact closed form (a constant logit offset makes
crossing age monotone in a single index), so the simulated VG20 understood
numbers must reproduce ``age_at_word_count.py``'s exact ones.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from scipy.special import expit

from vocab_growth import comparison as C
from vocab_growth import environment as env
from vocab_growth.models.definitions import MODEL_REGISTRY

TARGETS = [50, 150, 300, 500, 750]
SHARES = [0.50, 0.75, 0.90]
N_CHILDREN = 2000
DRAW_SUBSAMPLE = 1000
HDI_PROB = 0.89
CAPS = {"understood": 72.0, "spoken": 90.0}


def load(model: str) -> dict:
    """Latent population curves plus whatever child-effect scales the model has."""
    slope = "tau_subj_u_0" in _posterior_names(model)
    scal = (
        ("tau_subj_u_0", "tau_subj_u_1", "tau_subj_u_rho",
         "tau_subj_q_0", "tau_subj_q_1", "tau_subj_q_rho")
        if slope
        else ("tau_subj_u", "tau_subj_q", "rho_uq")
    )
    ages, (f_u, h), s = C._load_reshaped_draws(
        C.trace_path(model), ("f_u_plot", "h_plot"), scal
    )
    return {
        "model": model,
        "slope": slope,
        "ages": ages,
        "f_u": f_u,
        "h": h,
        "scal": s,
        "n_trials": C.n_trials(model),
        "ref_age": getattr(MODEL_REGISTRY[model], "subject_slope_ref_age_months", None),
    }


def _posterior_names(model: str) -> set[str]:
    import xarray as xr

    ds = xr.open_dataset(C.trace_path(model), group="posterior", engine="h5netcdf")
    names = set(ds.data_vars)
    ds.close()
    return names


def _corr_pair(rng, rho: float, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Two standard normals with correlation ``rho``."""
    a = rng.standard_normal(n)
    b = rho * a + np.sqrt(max(1.0 - rho**2, 0.0)) * rng.standard_normal(n)
    return a, b


def offsets(d: dict, i: int, rng, ages: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Child offsets on the understood and q logits: (children, ages) each.

    VG20's are constant in age and correlated across outcomes; VG19's vary with
    age and are correlated only within an outcome.
    """
    s = d["scal"]
    if not d["slope"]:
        zu, zq = _corr_pair(rng, float(s["rho_uq"][i]), N_CHILDREN)
        return (
            (s["tau_subj_u"][i] * zu)[:, None] * np.ones_like(ages)[None, :],
            (s["tau_subj_q"][i] * zq)[:, None] * np.ones_like(ages)[None, :],
        )
    D = ((ages - d["ref_age"]) / 12.0)[None, :]
    out = []
    for tag in ("u", "q"):
        t0 = float(s[f"tau_subj_{tag}_0"][i])
        t1 = float(s[f"tau_subj_{tag}_1"][i])
        r = float(s[f"tau_subj_{tag}_rho"][i])
        a, b = _corr_pair(rng, r, N_CHILDREN)
        out.append((t0 * a)[:, None] + (t1 * b)[:, None] * D)
    return out[0], out[1]


def crossing_ages(Y: np.ndarray, ages: np.ndarray, level: float) -> np.ndarray:
    """First crossing, with 'never reaches' as +inf and 'before the grid' as -inf."""
    out = C.first_crossing_age(Y, ages, level)
    out = np.where(np.isnan(out) & (Y[:, 0] > level), -np.inf, out)
    return np.where(np.isnan(out), np.inf, out)


def summarise(values: np.ndarray, cap: float) -> dict:
    median = float(np.median(values))
    finite = np.isfinite(values)
    lo, hi = (
        C.hdi_from_samples(values[finite], HDI_PROB) if finite.sum() > 1 else (np.nan, np.nan)
    )
    ok = np.isfinite(median) and median <= cap
    return {
        "median": median if ok else np.nan,
        "lo": float(lo) if np.isfinite(lo) else np.nan,
        "hi": float(hi) if np.isfinite(hi) else np.nan,
        "frac_beyond_window": float(np.mean(values > cap)),
        "beyond_cap": not ok,
    }


def run(model: str) -> pd.DataFrame:
    d = load(model)
    ages, f_u, h, n = d["ages"], d["f_u"], d["h"], d["n_trials"]
    rng = np.random.default_rng(20260823)
    idx = rng.choice(f_u.shape[0], size=min(DRAW_SUBSAMPLE, f_u.shape[0]), replace=False)
    keys = [(o, t, s) for o in CAPS for t in TARGETS for s in SHARES]
    acc = {k: np.full(len(idx), np.nan) for k in keys}

    for k, i in enumerate(idx):
        off_u, off_q = offsets(d, i, rng, ages)
        U = n * expit(f_u[i][None, :] + off_u)
        S = U * expit(h[i][None, :] + off_q)
        for outcome, Y in (("understood", U), ("spoken", S)):
            for target in TARGETS:
                cross = crossing_ages(Y, ages, target)
                for share in SHARES:
                    acc[(outcome, target, share)][k] = np.quantile(
                        cross, share, method="inverted_cdf"
                    )
    rows = [
        {"model": model.upper(), "outcome": o, "target": t, "share": s,
         **summarise(acc[(o, t, s)], CAPS[o])}
        for (o, t, s) in keys
    ]
    return pd.DataFrame(rows)


def table(df: pd.DataFrame, outcome: str) -> None:
    cap = CAPS[outcome]
    models = list(dict.fromkeys(df.model))
    print(f"\n=== Words {outcome} — age (months) by which each share reaches the count ===")
    print(f"    reporting window ends at {cap:.0f} months")
    head = f"{'words':>6} {'model':>6} | " + " | ".join(
        f"{int(s * 100)}% of children".center(22) for s in SHARES
    )
    print(head)
    print("-" * len(head))
    for target in TARGETS:
        for m in models:
            r0 = df[(df.model == m) & (df.outcome == outcome) & (df.target == target)]
            cells = []
            for share in SHARES:
                r = r0[r0.share == share].iloc[0]
                if r["beyond_cap"]:
                    cells.append(f"not by {cap:.0f} mo".center(22))
                else:
                    hi = f"{r['hi']:.1f}" if np.isfinite(r["hi"]) and r["hi"] <= cap else f">{cap:.0f}"
                    cells.append(f"{r['median']:5.1f}  [{r['lo']:.1f}, {hi}]".center(22))
            print(f"{target:>6} {m:>6} | " + " | ".join(cells))
        print()


def main() -> None:
    frames = []
    for model in ("vg20", "vg19"):
        d = load(model)
        s = d["scal"]
        if d["slope"]:
            print(
                f"{model.upper()}  slope model, ref age {d['ref_age']:.0f} mo | "
                f"tau_u0={s['tau_subj_u_0'].mean():.3f} tau_u1={s['tau_subj_u_1'].mean():.3f} "
                f"rho_u={s['tau_subj_u_rho'].mean():+.3f} | "
                f"tau_q0={s['tau_subj_q_0'].mean():.3f} tau_q1={s['tau_subj_q_1'].mean():.3f} "
                f"rho_q={s['tau_subj_q_rho'].mean():+.3f}"
            )
        else:
            print(
                f"{model.upper()}  constant-offset model | "
                f"tau_u={s['tau_subj_u'].mean():.3f} tau_q={s['tau_subj_q'].mean():.3f} "
                f"rho_uq={s['rho_uq'].mean():+.3f}"
            )
        frames.append(run(model))
    df = pd.concat(frames, ignore_index=True)
    os.makedirs(env.comparisons_output_dir(), exist_ok=True)
    out = os.path.join(
        env.comparisons_output_dir(), "age_at_word_count_vg20_vg19.csv"
    )
    df.to_csv(out, index=False)
    for outcome in ("understood", "spoken"):
        table(df, outcome)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
