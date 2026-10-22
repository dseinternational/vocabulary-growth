#!/usr/bin/env python
# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Which reported quantities depend on the child-effect structure, and where?

VG10, VG19 and VG20 use independent constant child offsets, per-outcome
intercepts and slopes, and correlated constant offsets, respectively. The
registered VG20 also includes sex, so comparing these models does not isolate
the child-effect structure alone. Differences between compatible fitted curves
describe sensitivity to the combined specifications, separately from each
model's posterior interval width.

The yardstick is **VG20's own 89% ETI width** at the same age, because a
difference can be read against the uncertainty already shown. A gap near 1.0
means the two point estimates sit almost a full reference interval apart. This
descriptive ratio is not a test of equivalence or a decision threshold.

Also prints the pool's observation counts by age band, because the two are
read together: divergence where data are thin warrants checking both model
assumptions and the supported reporting range.

Cited by ``notes/202608221200-reporting-source-by-quantity.md``.
"""

import os

import numpy as np
import pandas as pd

from vocab_growth import data_utils as du
from vocab_growth import environment as env

BASE = os.path.join(env.output_root(), "models")
MODELS = {
    "VG10": "VG10-age-understood-spoken-ds-re-subj-uq-anchored",
    "VG19": "VG19-age-understood-spoken-ds-re-subj-uq-anchored-slope",
    "VG20": "VG20-age-understood-spoken-ds-re-subj-uq-anchored-corr",
}
AGES = (18, 24, 36, 48, 60, 72, 84)
REFERENCE = "VG20"


def _at(df, age, col):
    x = df["age_months"].to_numpy()
    return float(df[col].to_numpy()[np.abs(x - age).argmin()])


def compare(filename, prefix, scale, title):
    d = {k: pd.read_csv(os.path.join(BASE, v, filename)) for k, v in MODELS.items()}
    if f"{prefix}_median" not in d[REFERENCE].columns:
        print(f"  [{title}] no column {prefix}_median")
        return
    others = [m for m in MODELS if m != REFERENCE]
    print(f"\n=== {title} (x{scale}); gaps as a fraction of {REFERENCE}'s 89% ETI width")
    head = f"{'age':>4} " + "".join(f"{m:>10}" for m in MODELS)
    head += " |" + "".join(f"{m + '-' + REFERENCE:>14}{'/w':>6}" for m in others)
    print(head)
    worst = 0.0
    for age in AGES:
        v = {m: _at(d[m], age, f"{prefix}_median") for m in MODELS}
        w = (_at(d[REFERENCE], age, f"{prefix}_ci_hi")
             - _at(d[REFERENCE], age, f"{prefix}_ci_lo")) * scale
        row = f"{age:4d} " + "".join(f"{v[m] * scale:10.2f}" for m in MODELS) + " |"
        for m in others:
            gap = (v[m] - v[REFERENCE]) * scale
            frac = abs(gap) / w if w > 0 else float("nan")
            if m == "VG19":
                worst = max(worst, frac)
            row += f"{gap:+14.2f}{frac:6.2f}"
        print(row)
    print(f"     worst VG19 gap: {worst:.2f} interval widths")


def age_bands():
    d = du.load_combined_data()
    print("\n=== pool density by age band (why the divergence sits where it does)")
    print(f"{'band':>10} {'rows':>6} {'children':>9} {'understood':>11} {'spoken':>7}")
    for lo, hi in ((8, 24), (24, 36), (36, 48), (48, 60), (60, 72), (72, 84), (84, 120)):
        m = (d.age >= lo) & (d.age < hi)
        child = d[m].study.astype(str) + "|" + d[m].subject_id.astype(str)
        print(f"{lo:3d}-{hi:<3d}  {int(m.sum()):6d} {child.nunique():9d} "
              f"{int(d[m].understood.notna().sum()):11d} {int(d[m].spoken.notna().sum()):7d}")


if __name__ == "__main__":
    compare("posterior_summary_u.csv", "p", 810, "understood, population curve")
    compare("posterior_summary_q.csv", "q", 1, "production ratio q")
    compare("posterior_summary_s.csv", "p_population", 810, "spoken, POPULATION curve")
    compare("posterior_summary_s.csv", "p_subject_marginal", 810,
            "spoken, SUBJECT-MARGINAL curve")
    age_bands()
