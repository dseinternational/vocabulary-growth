#!/usr/bin/env python
# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Persist the pool counts the report quotes in prose.

The Discussion's Limitations section describes who the data describe and where
coverage thins, in sentences carrying a dozen counts. They were typed, and by
the #215 data build every one of the pool-level figures had drifted: 1,638
administrations from 846 children against an actual 1,521 from 781, and the
per-band coverage counts with them. Nothing recomputed them because they are
*data* counts rather than fit outputs, so no artefact carried them and no check
could compare.

This writes them where the report can read them at render time, on the same
principle as every other number in the report. It reads the prepared DuckDB
build through the ordinary loader, so it sees exactly the pool the models see,
including the documented masking.

Writes ``<comparisons>/pool_descriptives.csv`` (one row) and
``<comparisons>/pool_coverage_bands.csv`` (one row per 6-month band).

Usage::

    python scripts/pool_descriptives.py
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from vocab_growth import data_utils as du  # noqa: E402
from vocab_growth import environment as env  # noqa: E402
from vocab_growth.reporting import heading  # noqa: E402

SUMMARY_FILENAME = "pool_descriptives.csv"
BANDS_FILENAME = "pool_coverage_bands.csv"
BAND_WIDTH = 6

# Study prefixes map to the countries the Limitations section names.
COUNTRIES = {"es": "Spain", "ie": "Ireland", "it": "Italy",
             "nz": "New Zealand", "uk": "United Kingdom", "us": "United States"}


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    repeats = df.groupby("subject_id").size()
    signed = df[df["signed"].notna()]
    countries = sorted({COUNTRIES.get(s.split("_")[0], s.split("_")[0])
                        for s in df["study"].astype(str)})
    return pd.DataFrame([{
        "administrations": len(df),
        "children": int(df["subject_id"].nunique()),
        "studies": int(df["study"].nunique()),
        "countries": len(countries),
        "country_list": "; ".join(countries),
        "children_with_repeats": int((repeats > 1).sum()),
        "children_single_observation": int((repeats == 1).sum()),
        "understood_rows": int(df["understood"].notna().sum()),
        "spoken_rows": int(df["spoken"].notna().sum()),
        "signed_rows": int(df["signed"].notna().sum()),
        "signed_children": int(signed["subject_id"].nunique()),
        "signed_studies": int(signed["study"].nunique()),
        "signed_max_age": float(signed["age"].max()) if len(signed) else np.nan,
        "age_min": float(df["age"].min()),
        "age_max": float(df["age"].max()),
    }])


def coverage_bands(df: pd.DataFrame) -> pd.DataFrame:
    lo_edge = int(np.floor(df["age"].min() / BAND_WIDTH) * BAND_WIDTH)
    hi_edge = int(np.ceil(df["age"].max() / BAND_WIDTH) * BAND_WIDTH)
    rows = []
    for lo in range(lo_edge, hi_edge, BAND_WIDTH):
        band = df[(df["age"] >= lo) & (df["age"] < lo + BAND_WIDTH)]
        rows.append({
            "age_lo": lo,
            "age_hi": lo + BAND_WIDTH,
            "administrations": len(band),
            "understood": int(band["understood"].notna().sum()),
            "spoken": int(band["spoken"].notna().sum()),
            "signed": int(band["signed"].notna().sum()),
            "studies": int(band["study"].nunique()),
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    env.set_output_root(args.output_dir)
    out_dir = env.comparisons_output_dir()
    os.makedirs(out_dir, exist_ok=True)

    heading("Pool descriptives", style="bold cyan")
    df = du.load_combined_data()
    summary = summarise(df)
    bands = coverage_bands(df)
    summary.to_csv(os.path.join(out_dir, SUMMARY_FILENAME), index=False)
    bands.to_csv(os.path.join(out_dir, BANDS_FILENAME), index=False)

    row = summary.iloc[0]
    print(f"  {row['administrations']} administrations from {row['children']} children "
          f"in {row['studies']} studies across {row['countries']} countries")
    print(f"  {row['children_with_repeats']} children assessed more than once")
    print(f"  signed: {row['signed_rows']} administrations, {row['signed_children']} "
          f"children, {row['signed_studies']} studies, to {row['signed_max_age']:.0f} months")
    print(f"\nwrote {SUMMARY_FILENAME} and {BANDS_FILENAME} to {out_dir}")


if __name__ == "__main__":
    main()
