# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Does early signing predict later *spoken* vocabulary, beyond comprehension?

The practice question behind the signing results: children with Down syndrome who
sign early — do they go on to say more than otherwise-similar children who did
not? This script measures the association and, more importantly, measures how far
the data can support an answer at all.

Design
------
One row per child. At the **early wave** ``t0`` — the first administration
carrying signed, understood *and* spoken — we record the child's signing, their
comprehension standing and their existing speech. At the **latest later wave**
``t1`` we record spoken vocabulary. Every measure is an age- and study-adjusted
logit residual, the same scoring
``scripts/experiments/rank_stability.py`` uses, so "standing" means the same
thing here as it does there.

The estimand is the coefficient on early signing in

    resid_spoken(t1) ~ resid_spoken(t0) + resid_understood(t0) + signing(t0)

Conditioning on ``resid_spoken(t0)`` is the analytical heart of it. Signing is
taught *because* a child is not talking, so early signing marks low speech, and a
model that omits prior speech measures that selection rather than any effect of
signing. With prior speech held, the coefficient asks the narrower and more
answerable question: among children at the same comprehension standing and the
same starting speech level, is signing associated with more speech later?

Two signing measures, because they answer different questions:

* ``signs`` — binary, "does this child sign at all". The practitioner framing.
* ``sign_dose`` — the age- and study-adjusted residual of the signed fraction of
  comprehension. Retains within-study contrast where the binary has almost none.

What this cannot be
-------------------
**Not causal, and the confounding is unusually severe rather than pro forma.**
Signing is not randomly assigned; it is a decision taken by families and by the
programmes the studies recruit from. §"design" in the output shows the damage:
signing status is very nearly a function of *study*, with two studies at ~100%
signers and one at 0%. So a study-adjusted estimate rests on the three studies
that actually vary internally, and an unadjusted one is mostly a between-study
comparison wearing a within-child disguise. Both are reported, and they should be
read as a range bracketing the design's ambiguity rather than as an estimate and
a robustness check.

The residual selection runs one way, which is the one thing here that helps: any
remaining indication bias makes early signers look *worse*, so a positive
coefficient is harder to manufacture than a negative one.

Usage::

    python scripts/experiments/early_signing_and_later_speech.py
    python scripts/experiments/early_signing_and_later_speech.py --n-boot 4000
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from vocab_growth import data_utils as du

RNG = np.random.default_rng(20260816)
MIN_STUDY_N = 8
#: A study contributes identifying variation only if it is not all-or-nothing.
MIN_WITHIN_STUDY_SIGNERS = 3


# ----------------------------------------------------------------- scoring


def _adjust(frame: pd.DataFrame, value: np.ndarray, ceiling: np.ndarray) -> np.ndarray:
    """Age- and study-adjusted logit residual for one outcome column.

    Cubic in standardised age plus study dummies, matching
    ``rank_stability.adjusted_scores``. The clip at half an item keeps zero and
    ceiling counts finite — 16% of the conditional rows have zero spoken words,
    and an unclipped ``logit(0)`` would put a large negative outlier on exactly
    the small-vocabulary children this analysis is about.
    """
    p = np.clip(value / ceiling, 0.5 / ceiling, 1 - 0.5 / ceiling)
    logit = np.log(p / (1 - p))
    a = (frame.age.values - frame.age.values.mean()) / frame.age.values.std()
    dummies = pd.get_dummies(frame.study, drop_first=True).values.astype(float)
    X = np.column_stack([np.ones(len(a)), a, a**2, a**3, dummies])
    beta, *_ = np.linalg.lstsq(X, logit, rcond=None)
    return logit - X @ beta


def build_frame() -> pd.DataFrame:
    """One row per child: early signing/comprehension/speech, later speech."""
    d = du.load_combined_data()
    d = d[d.survey_vocab_max.notna()].sort_values(["subject_id", "age"])

    rows = []
    for _sid, ch in d.groupby("subject_id"):
        ch = ch.sort_values("age")
        complete = ch[ch.signed.notna() & ch.understood.notna() & ch.spoken.notna()]
        if complete.empty:
            continue
        t0 = complete.iloc[0]
        later = ch[(ch.age > t0.age) & ch.spoken.notna()]
        if later.empty:
            continue
        t1 = later.iloc[-1]
        rows.append({
            "study": ch.study.iloc[0],
            "child": f"{ch.study.iloc[0]}|{ch.subject_id.iloc[0]}",
            "age0": float(t0.age), "age1": float(t1.age),
            "gap": float(t1.age - t0.age),
            "signed0": float(t0.signed), "understood0": float(t0.understood),
            "spoken0": float(t0.spoken), "spoken1": float(t1.spoken),
            "ceiling0": float(t0.survey_vocab_max),
            "ceiling1": float(t1.survey_vocab_max),
        })
    f = pd.DataFrame(rows)

    f["r_und0"] = _adjust(f.rename(columns={"age0": "age"}), f.understood0.values, f.ceiling0.values)
    f["r_sp0"] = _adjust(f.rename(columns={"age0": "age"}), f.spoken0.values, f.ceiling0.values)
    f["r_sp1"] = _adjust(
        f.assign(age=f.age1), f.spoken1.values, f.ceiling1.values
    )
    # Signing dose: the signed fraction of comprehension, adjusted the same way.
    share = np.clip(f.signed0.values / np.maximum(f.understood0.values, 1.0), 0.0, 1.0)
    f["sign_share"] = share
    f["sign_dose"] = _adjust(
        f.rename(columns={"age0": "age"}),
        share * f.ceiling0.values,
        f.ceiling0.values,
    )
    f["signs"] = (f.signed0 > 0).astype(float)
    return f


# ----------------------------------------------------------------- estimation


def _ols(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta


def _design(f: pd.DataFrame, predictor: str, *, study_fe: bool) -> np.ndarray:
    cols = [np.ones(len(f)), f.r_sp0.values, f.r_und0.values, f[predictor].values,
            f.gap.values, f.age0.values]
    if study_fe:
        cols.append(pd.get_dummies(f.study, drop_first=True).values.astype(float))
    return np.column_stack([c if c.ndim > 1 else c.reshape(-1, 1) for c in cols])


def estimate(f: pd.DataFrame, predictor: str, *, study_fe: bool, n_boot: int) -> dict:
    """Coefficient on ``predictor`` with a child-level bootstrap interval.

    Reported **per SD of the predictor**, so the binary and dose measures are on
    one scale and both are comparable with the tracking note's between-child
    units. The raw coefficient is kept too, since the binary one is already
    interpretable as signer-versus-not.
    """
    scale = float(np.std(f[predictor].values, ddof=1))
    X = _design(f, predictor, study_fe=study_fe)
    point = _ols(f.r_sp1.values, X)[3]

    draws = []
    idx = np.arange(len(f))
    for _ in range(n_boot):
        take = RNG.choice(idx, size=len(idx), replace=True)
        g = f.iloc[take]
        try:
            b = _ols(g.r_sp1.values, _design(g, predictor, study_fe=study_fe))[3]
        except np.linalg.LinAlgError:  # pragma: no cover - degenerate resample
            continue
        draws.append(b * float(np.std(g[predictor].values, ddof=1)))
    lo, hi = np.percentile(draws, [5.5, 94.5])
    return {"beta_raw": float(point), "beta": float(point * scale),
            "lo": float(lo), "hi": float(hi),
            "p_gt0": float(np.mean(np.array(draws) > 0)), "n": int(len(f))}


def describe_design(f: pd.DataFrame) -> pd.DataFrame:
    """Per-study signing rates — the table that decides what is identifiable."""
    g = f.groupby("study").agg(
        children=("child", "size"),
        signers=("signs", "sum"),
        median_age0=("age0", "median"),
        median_gap=("gap", "median"),
    )
    g["signer_rate"] = g.signers / g.children
    g["identifying"] = (
        (g.signers >= MIN_WITHIN_STUDY_SIGNERS)
        & (g.children - g.signers >= MIN_WITHIN_STUDY_SIGNERS)
    )
    return g.sort_values("children", ascending=False)


def leave_one_study_out(f: pd.DataFrame, predictor: str, *, study_fe: bool) -> pd.DataFrame:
    """Per-SD coefficient with each study dropped in turn.

    With six studies and signing very nearly a study-level property, this is the
    check that matters more than the bootstrap: an association carried by one
    study is a study difference, not a child-level one.
    """
    rows = []
    for s in sorted(f.study.unique()):
        g = f[f.study != s]
        if len(g) < MIN_STUDY_N:
            continue
        X = _design(g, predictor, study_fe=study_fe)
        b = _ols(g.r_sp1.values, X)[3] * float(np.std(g[predictor].values, ddof=1))
        rows.append({"dropped": s, "n": len(g), "beta_per_sd": float(b)})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------- report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()

    f = build_frame()
    print(f"\n=== FRAME: {len(f)} children with signed+understood+spoken at an early "
          f"wave and spoken at a later one ===")
    print(f"  age at t0: median {f.age0.median():.0f} mo "
          f"(range {f.age0.min():.0f}-{f.age0.max():.0f})")
    print(f"  gap t0->t1: median {f.gap.median():.0f} mo "
          f"(IQR {f.gap.quantile(.25):.0f}-{f.gap.quantile(.75):.0f})")
    print(f"  signers at t0: {int(f.signs.sum())} of {len(f)}")

    design = describe_design(f)
    print("\n--- design: is signing separable from study? ---")
    print(design.to_string(float_format=lambda v: f"{v:.2f}"))
    ident = design[design.identifying]
    n_ident = int(design.loc[design.identifying, "children"].sum())
    print(f"\n  studies with within-study variation in signing: "
          f"{len(ident)} of {len(design)}, carrying {n_ident} of {len(f)} children.")
    print("  The rest are all-signers or all-non-signers, so they identify the")
    print("  binary contrast only through the between-study comparison.")

    print("\n--- association with later spoken vocabulary ---")
    print("  (controls: prior spoken standing, prior comprehension standing, gap, age)")
    out = []
    for predictor in ("signs", "sign_dose"):
        for study_fe in (False, True):
            r = estimate(f, predictor, study_fe=study_fe, n_boot=args.n_boot)
            label = f"{predictor}{' + study FE' if study_fe else ''}"
            out.append({"model": label, **r})
            print(f"  {label:<22} beta/SD = {r['beta']:+.3f} "
                  f"[{r['lo']:+.3f}, {r['hi']:+.3f}]  P(>0) = {r['p_gt0']:.2f}"
                  f"   (raw {r['beta_raw']:+.3f})")

    for predictor in ("signs", "sign_dose"):
        print(f"\n--- leave-one-study-out (study FE, {predictor}) ---")
        loso = leave_one_study_out(f, predictor, study_fe=True)
        print(loso.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        span = loso.beta_per_sd
        print(f"  range across deletions: {span.min():+.3f} to {span.max():+.3f}"
              f"  ({'sign-stable' if span.min() * span.max() > 0 else 'SIGN FLIPS'})")

    print("\n--- what would change the answer ---")
    print("  Nothing here is randomised. Signing is taught because a child is not")
    print("  talking, so the residual selection pushes the coefficient DOWN: a")
    print("  positive estimate survives that bias, a negative one is confounded")
    print("  with it and cannot be separated in these data.")
    return pd.DataFrame(out)


if __name__ == "__main__":
    main()
