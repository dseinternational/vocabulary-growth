# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Compare VG20 with VG10 using the screening criteria from issue #224.

VG20 adds correlation between the child's comprehension and production-ratio
effects. At fixed values of the other parameters, this leaves the reference
curves and each effect's marginal Normal distribution unchanged. It can change
the distribution of spoken vocabulary, which is the product of comprehension
and the production ratio.

The script checks three empirical criteria:

1. VG20 reference-curve medians lie inside VG10's 89% intervals at each age.
2. The median ratio of understood subject-marginal interval widths is within
   2% of one.
3. The median ratio of spoken subject-marginal interval widths exceeds 1.02.

These are screening rules for separately fitted posteriors. Adding correlation
can change other parameter estimates, so a failed rule does not by itself prove
an implementation error. ``Ey_population`` uses zero child and study effects;
it is not the average after integrating over those effects.

Usage::

    python scripts/compare_vg10_vg20.py [--output-dir DIR]
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from vocab_growth import environment as env

BASE = ("vg10", "VG10-age-understood-spoken-ds-re-subj-uq-anchored")
VARIANT = ("vg20", "VG20-age-understood-spoken-ds-re-subj-uq-anchored-corr")


def _read(model_dir: str, filename: str) -> pd.DataFrame:
    path = os.path.join(env.output_root(), "models", model_dir, filename)
    if not os.path.isfile(path):
        raise SystemExit(f"missing artefact: {path}")
    return pd.read_csv(path)


def _width(frame: pd.DataFrame, stem: str, side: str) -> pd.Series:
    """Outer-interval width, which is what criteria 2 and 3 compare."""
    return frame[f"{stem}_ci_hi_{side}"] - frame[f"{stem}_ci_lo_{side}"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    env.set_output_root(args.output_dir)

    rows: list[dict] = []

    # -- Criterion 1: population-level must not move ------------------------
    for outcome, filename in (("understood", "posterior_summary_u.csv"),
                              ("spoken", "posterior_summary_s.csv")):
        b = _read(BASE[1], filename)
        v = _read(VARIANT[1], filename)
        merged = b.merge(v, on="age_months", suffixes=("_base", "_var"))
        inside = (
            (merged["Ey_population_median_var"] >= merged["Ey_population_ci_lo_base"])
            & (merged["Ey_population_median_var"] <= merged["Ey_population_ci_hi_base"])
        )
        delta = merged["Ey_population_median_var"] - merged["Ey_population_median_base"]
        rows.append({
            "criterion": "1 population-level unchanged",
            "quantity": f"Ey_population[{outcome}]",
            "expect": "no movement",
            "n_ages": len(merged),
            "n_inside_base_ci": int(inside.sum()),
            "max_abs_delta": float(np.abs(delta).max()),
            "pass": bool(inside.all()),
        })

    b = _read(BASE[1], "production_rate.csv")
    v = _read(VARIANT[1], "production_rate.csv")
    merged = b.merge(v, on="age_months", suffixes=("_base", "_var"))
    inside = (
        (merged["q_median_var"] >= merged["ci_lo_base"])
        & (merged["q_median_var"] <= merged["ci_hi_base"])
    )
    rows.append({
        "criterion": "1 population-level unchanged",
        "quantity": "q",
        "expect": "no movement",
        "n_ages": len(merged),
        "n_inside_base_ci": int(inside.sum()),
        "max_abs_delta": float(np.abs(merged["q_median_var"] - merged["q_median_base"]).max()),
        "pass": bool(inside.all()),
    })

    # -- Criteria 2 and 3: subject-marginal spread --------------------------
    # These are empirical width criteria, not identities between refitted models.
    for outcome, filename, expect_widening in (
        ("understood", "posterior_summary_u.csv", False),
        ("spoken", "posterior_summary_s.csv", True),
    ):
        b = _read(BASE[1], filename)
        v = _read(VARIANT[1], filename)
        merged = b.merge(v, on="age_months", suffixes=("_base", "_var"))
        w_base = _width(merged, "Ey_subject_marginal", "base")
        w_var = _width(merged, "Ey_subject_marginal", "var")
        ratio = (w_var / w_base).replace([np.inf, -np.inf], np.nan).dropna()
        median_ratio = float(ratio.median())
        # Use the prespecified 2% tolerance; it is not an estimated Monte Carlo error.
        if expect_widening:
            passed = median_ratio > 1.02
            criterion = "3 spoken subject-marginal widens"
        else:
            passed = 0.98 <= median_ratio <= 1.02
            criterion = "2 understood subject-marginal unchanged"
        rows.append({
            "criterion": criterion,
            "quantity": f"Ey_subject_marginal width[{outcome}]",
            "expect": "widen" if expect_widening else "no change",
            "n_ages": len(merged),
            "n_inside_base_ci": "",
            "max_abs_delta": float(np.abs(w_var - w_base).max()),
            "pass": bool(passed),
            "median_width_ratio": round(median_ratio, 4),
        })

    table = pd.DataFrame(rows)
    out_dir = os.path.join(env.output_root(), "comparisons")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "vg10_vs_vg20_gate3.csv")
    table.to_csv(path, index=False)
    print(table.to_string(index=False))
    print(f"\nwritten: {path}")
    print(
        "\nGATE 3 "
        + ("PASSES" if bool(table["pass"].all()) else "FAILS")
        + " — remember criterion 3 passes by CHANGING; only 1 and 2 pass by staying put."
    )


if __name__ == "__main__":
    main()
