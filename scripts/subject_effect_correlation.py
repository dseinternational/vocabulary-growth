#!/usr/bin/env python
# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Summarise correlation across fitted child effects, one value per draw.

Read comprehension and production-ratio effects from a fitted model. On VG09,
VG10 and VG16 the priors are independent, but fitted effects may correlate through
the data. On VG20 compare their realised correlation with the population
parameter ``rho_uq``. They need not agree because the realised children form a
finite sample and their effects carry unequal information and shrinkage.

This is not a lower bound on the population correlation. Write the posterior
summary to ``<comparisons>/ds_subject_effect_correlation.csv``.

Usage::

    python scripts/subject_effect_correlation.py
    python scripts/subject_effect_correlation.py vg20 vg10
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
from vocab_growth.comparisons_provenance import (  # noqa: E402
    ComparisonOutputs,
    write_comparison_manifest,
)
from vocab_growth.fit_consumers import (  # noqa: E402
    add_allow_stale_argument,
    contributing_fits,
)
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
    parser.add_argument("models", nargs="*", default=["vg20"], help="model keys")
    parser.add_argument("--thin", type=int, default=20)
    parser.add_argument("--output-dir", default=None)
    add_allow_stale_argument(parser)
    args = parser.parse_args()

    env.set_output_root(args.output_dir)
    out_dir = env.comparisons_output_dir()
    os.makedirs(out_dir, exist_ok=True)

    heading("Subject-effect correlation", style="bold cyan")
    keys = args.models or ["vg20"]
    # Every fit this comparison reads is checked against the registered
    # definition and the current prepared frame, and recorded in the
    # comparisons manifest so the outputs cannot outlive a refit unnoticed
    # (issue #266 findings 1 and 7).
    contributing = contributing_fits(
        keys,
        consumer="subject_effect_correlation.py",
        allow_stale=args.allow_stale_fit,
    )
    written = ComparisonOutputs(out_dir)
    rows = [summarise(key, args.thin) for key in keys]
    frame = pd.DataFrame(rows)
    path = os.path.join(out_dir, FILENAME)
    frame.to_csv(path, index=False)
    write_comparison_manifest(
        out_dir,
        script="subject_effect_correlation.py",
        contributing=contributing,
        outputs=written.written(),
        arguments=keys,
    )
    for row in rows:
        print(
            f"  {row['model']}: corr {row['corr_median']:+.3f} "
            f"(89% {row['corr_ci_lo']:+.3f} to {row['corr_ci_hi']:+.3f}), "
            f"P(>0) = {row['corr_p_gt0']:.3f}, {row['n_draws']} draws"
        )
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
