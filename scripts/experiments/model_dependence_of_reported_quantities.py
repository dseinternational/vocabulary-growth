#!/usr/bin/env python
# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Which reported quantities depend on the child-effect structure, and where?

VG10, VG19 and VG20 differ only in how a child departs from the population
trajectory -- independent constant offsets, per-outcome offset **and rate**, and
correlated constant offsets respectively. Everything else in the three graphs is
identical. So the difference between their fitted curves at a given age is a
direct measurement of how much a reported number depends on a modelling choice
the data does not settle, and that is a different question from how wide any one
model's credible interval is.

The yardstick is **VG20's own 89% ETI width** at the same age, because a
difference only matters to a reader in proportion to the uncertainty already
being shown. A gap of 0.1 interval widths is invisible; a gap near 1.0 means the
two models' point estimates sit almost a full published interval apart.

Also prints the pool's observation counts by age band, because the two are
read together: divergence concentrated where the data are thin is a statement
about the reporting range, not about the models.

Cited by ``notes/202608221200-reporting-source-by-quantity.md``.
"""

import numpy as np
import pandas as pd

from vocab_growth import data_utils as du

BASE = "/scratch2/vg-output/models/"
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
    d = {k: pd.read_csv(BASE + v + "/" + filename) for k, v in MODELS.items()}
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
