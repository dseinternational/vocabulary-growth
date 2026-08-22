# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Gate 3 of #224: VG20 against VG10, with the pass criteria fixed in advance.

VG10 is VG20 at ``rho_uq = 0``, so the two are nested and most of what they
report must agree. The trap this script exists to avoid is stated in #224's own
comment thread: the issue says "the reported trajectories should be unchanged"
and treats movement as a red flag, but a positive ``rho_uq`` **should** move one
family of quantities. Run without that distinction drawn first, a correct result
reads as a failure.

So the criteria are declared here, before any VG20 fit exists, and each one names
the direction it expects rather than only a tolerance:

1. **Population-level quantities must not move.** ``Ey_population`` on both
   outcomes and the production ratio ``q`` describe the average child, and the
   correlation between two mean-zero deviates does not change their mean. Each
   VG20 value must sit inside VG10's own 89% interval at every reported age --
   the same standard ``compare_sensitivity.py`` applies to a prior variant.

2. **Understood subject-marginal spread must not move either.** This is the
   sharp one. The Cholesky construction sets
   ``delta_q = tau_q (rho z1 + sqrt(1 - rho^2) z2)``, which changes how the two
   deviates co-vary while leaving each one's marginal SD exactly as it was. So a
   correlation cannot widen comprehension on its own, and if it does, the
   whitening term is wrong and ``tau_subj_q`` has been silently rescaled -- a
   defect that would otherwise show up only as a slightly different number in a
   reported quantity.

3. **Spoken subject-marginal spread SHOULD widen.** Spoken is ``p_U * q``, so a
   positive correlation compounds: a child above average on comprehension tends
   also to convert more of it to speech. That is the whole motivation in #224 --
   VG10's independent draws understate how much children with Down syndrome
   differ from one another in speech. Widening here is the correction working,
   not a red flag. Narrowing, or no change at all, is the failure.

Criterion 3 is therefore the only one whose *pass* is a change. Reporting it
alongside 1 and 2 is what makes the gate readable.

Usage:
    python scripts/compare_vg10_vg20.py [--output-dir <dir>]
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
    # Understood must not widen (the correlation preserves each marginal SD);
    # spoken should, because spoken is the product of the two deviates.
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
        # 2% either way is the "unchanged" band: these are Monte Carlo interval
        # bounds from independent chains, so exact equality is not available.
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
