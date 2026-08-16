#!/usr/bin/env python
# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Measure the correlation the joint DS models assume away.

VG10 gives each child a comprehension deviation and a production-ratio
deviation, drawn independently, and the DS spoken between-child scale reported
in the DS-vs-TD ``tau`` contrast is derived on that independence. The TD
comparator (VG11) places one intercept on the spoken logit and assumes nothing
of the kind, so the assumption is an asymmetry in the contrast itself, not an
internal modelling detail.

Nothing in the model estimates the correlation, so this measures it in the
fitted per-child deviations and writes it where the reports can read it. The
number is not a free parameter recovered from the data — it is shrunk toward
zero by the independence prior — so it bounds the magnitude rather than
estimating it, which is what the reports say when they quote it.

Writes ``<comparisons>/ds_subject_effect_correlation.csv``::

    model,n_children,n_draws,corr_median,corr_ci50_lo,corr_ci50_hi,
    corr_ci_lo,corr_ci_hi,corr_p_gt0

Usage::

    python scripts/subject_effect_correlation.py            # vg10
    python scripts/subject_effect_correlation.py vg10 vg09
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from vocab_growth import comparison as C  # noqa: E402
from vocab_growth import environment as env  # noqa: E402
from vocab_growth import intervals  # noqa: E402
from vocab_growth.reporting import heading  # noqa: E402

FILENAME = "ds_subject_effect_correlation.csv"


def summarise(key: str, thin: int) -> dict:
    r, n_children = C.subject_effect_correlation(key, thin=thin)
    r = r[~np.isnan(r)]
    if r.size == 0:
        raise ValueError(f"{key}: no usable draws for the subject-effect correlation.")
    lo50, hi50 = intervals.interval_1d(r, intervals.INNER_CI_PROB, "eti")
    lo89, hi89 = intervals.interval_1d(r, intervals.DEFAULT_CI_PROB, "eti")
    return {
        "model": key.upper(),
        "n_children": n_children,
        "n_draws": int(r.size),
        "corr_median": float(np.median(r)),
        "corr_ci50_lo": lo50,
        "corr_ci50_hi": hi50,
        "corr_ci_lo": lo89,
        "corr_ci_hi": hi89,
        "corr_p_gt0": float(np.mean(r > 0.0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", nargs="*", default=["vg10"], help="model keys")
    parser.add_argument("--thin", type=int, default=20)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    env.set_output_root(args.output_dir)
    out_dir = env.comparisons_output_dir()
    os.makedirs(out_dir, exist_ok=True)

    heading("Subject-effect correlation", style="bold cyan")
    rows = [summarise(key, args.thin) for key in (args.models or ["vg10"])]
    frame = pd.DataFrame(rows)
    path = os.path.join(out_dir, FILENAME)
    frame.to_csv(path, index=False)
    for row in rows:
        print(
            f"  {row['model']}: corr {row['corr_median']:+.3f} "
            f"(89% {row['corr_ci_lo']:+.3f} to {row['corr_ci_hi']:+.3f}), "
            f"P(>0) = {row['corr_p_gt0']:.3f}, {row['n_draws']} draws"
        )
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
